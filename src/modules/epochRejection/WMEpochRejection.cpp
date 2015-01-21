//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <list>
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

// Input & output data
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/util/profiler/WLTimeProfiler.h"

#include "WBadChannelManager.h"
#include "WBadEpochManager.h"
#include "WMEpochRejection.h"
#include "WEpochRejection.h"
#include "WEpochRejectionSingle.h"
#include "WEpochRejectionTotal.h"
#include "WThreshold.h"

#include "WMEpochRejection.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEpochRejection )

WMEpochRejection::WMEpochRejection()
{
}

WMEpochRejection::~WMEpochRejection()
{
}

boost::shared_ptr< WModule > WMEpochRejection::factory() const
{
    return boost::shared_ptr< WModule >( new WMEpochRejection() );
}

const char** WMEpochRejection::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEpochRejection::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Epoch Rejection";
}

const std::string WMEpochRejection::getDescription() const
{
    return "Checks the input values of each modality for defined thresholds. Module supports LaBP data types only!";
}

void WMEpochRejection::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEpochRejection::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    /* PropertyGroup: thresholds for the several modalities  */
    m_propGrpRejection = m_properties->addPropertyGroup( "Rejection Properties", "Rejection Properties", false );
    m_propGrpEpochBuff = m_properties->addPropertyGroup( "Epoch Buffer Properties", "Epoch Buffer Properties", false );
    m_propGrpThresholds = m_properties->addPropertyGroup( "Rejection Thresholds", "Rejection Thresholds", false );
    m_propGrpEpoch = m_properties->addPropertyGroup( "Processing Statistic", "Processing Statistic", false );

    /* Rejection algorithm */
    m_rejectionType = WItemSelection::SPtr( new WItemSelection() );

    boost::shared_ptr< WItemSelectionItemTyped< WEpochRejection::SPtr > > item;
    WEpochRejection::SPtr rejection;

    // totalRejection
    rejection.reset( new WEpochRejectionTotal() );
    item.reset(
                    new WItemSelectionItemTyped< WEpochRejection::SPtr >( rejection, "Total Channel Rejection",
                                    "Computes total channel rejection." ) );
    m_rejectionType->addItem( item );

    // singleRejection
    rejection.reset( new WEpochRejectionSingle() );
    item.reset(
                    new WItemSelectionItemTyped< WEpochRejection::SPtr >( rejection, "Single Channel Rejection",
                                    "Computes single channel rejection." ) );
    m_rejectionType->addItem( item );

    // getting the SelectorProperty from the list an add it to the properties
    m_rejectionTypeSelection = m_propGrpRejection->addProperty( "Rejection Type", "Choose a rejection type.",
                    m_rejectionType->getSelectorFirst(), boost::bind( &WMEpochRejection::callbackRejectionTypeChanged, this ) );
    m_rejectionTypeSelection->changed( true );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_rejectionTypeSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_rejectionTypeSelection );

    /* reset button */
    m_resetModule = m_propGrpRejection->addProperty( "Reset the module", "Reset", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_resetModule->changed( true );

    /* epoch buffer properties */
    m_epochBufferSize = m_propGrpEpochBuff->addProperty( "Epoch Buffer Size", "Epoch Buffer Size", 5 );
    m_applyBufferSize = m_propGrpEpochBuff->addProperty( "Apply Buffer Size", "Apply", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_applyBufferSize->changed( true );

    /* Rejection file properties */
    m_rejectFileStatus = m_propGrpThresholds->addProperty( "CFG file status", "CFG file status.", NO_FILE_LOADED );
    m_rejectFileStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_rejectFile = m_propGrpThresholds->addProperty( "CFG file", "Read a CFG file, which contains the thresholds.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_rejectFile->changed( true );

    /* Thresholds */
    m_eegThreshold = m_propGrpThresholds->addProperty( "EEG threshold", "Threshold for the EEG modality.", EEG_THRESHOLD );
    m_eogThreshold = m_propGrpThresholds->addProperty( "EOG threshold", "Threshold for the EOG modality.", EOG_THRESHOLD );
    m_megGradThreshold = m_propGrpThresholds->addProperty( "MEG gradiometer threshold",
                    "Threshold for the MEG gradiometer channels.", MEG_GRAD_THRESHOLD );
    m_megMagThreshold = m_propGrpThresholds->addProperty( "MEG magnetometer threshold",
                    "Threshold for the MEG magnetometer channels.", MEG_MAG_THRESHOLD );
    m_applyThresholds = m_propGrpThresholds->addProperty( "Apply Thresholds", "Apply", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_applyThresholds->changed( true );

    /* counters */
    m_epochCount = m_propGrpEpoch->addProperty( "Epoch Count", "Number of epochs.", 0 );
    m_epochCount->setPurpose( PV_PURPOSE_INFORMATION );

    m_epochCountValid = m_propGrpEpoch->addProperty( "Valid Epochs", "Number of not rejected epochs.", 0 );
    m_epochCountValid->setPurpose( PV_PURPOSE_INFORMATION );

    m_epochCountInValid = m_propGrpEpoch->addProperty( "Invalid Epochs", "Number of rejected epochs.", 0 );
    m_epochCountInValid->setPurpose( PV_PURPOSE_INFORMATION );

    m_badChannelCount = m_propGrpEpoch->addProperty( "Bad Channels", "Number of bad channels.", 0 );
    m_badChannelCount->setPurpose( PV_PURPOSE_INFORMATION );

    m_badEpochCount = m_propGrpEpoch->addProperty( "Rejected Epochs", "Number of rejected epochs in the buffer.", 0 );
    m_badEpochCount->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMEpochRejection::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    m_epochCount->set( 0, true );
    m_epochCountValid->set( 0, true );
    m_epochCountInValid->set( 0, true );
    m_badChannelCount->set( 0, true );
    m_badEpochCount->set( 0, true );

    m_parser.reset( new WThresholdParser() );

    // modality-property field -matching for parsing.
    m_modalityLabelMap.reset( new ModalityUIFiledMap() );
    m_modalityLabelMap->insert( ModalityUIFiledMap::value_type( WLEModality::EEG, m_eegThreshold ) );
    m_modalityLabelMap->insert( ModalityUIFiledMap::value_type( WLEModality::EOG, m_eogThreshold ) );
    m_modalityLabelMap->insert( ModalityUIFiledMap::value_type( WLEModality::MEG_GRAD, m_megGradThreshold ) );
    m_modalityLabelMap->insert( ModalityUIFiledMap::value_type( WLEModality::MEG_MAG, m_megMagThreshold ) );
    m_thresholds.reset( new WThreshold::WThreshold_List() );

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );

    callbackRejectionTypeChanged(); // set the rejection type.

    initThresholds();

    infoLog() << "Initializing module finished!";
}

void WMEpochRejection::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr emmIn;

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        // parsing a ".cfg" file
        if( m_rejectFile->changed( true ) )
        {
            m_rejectFileStatus->set( WMEpochRejection::LOADING_FILE, true ); // change file notification

            if( m_parser->parse( m_rejectFile->get().string() ) ) // start parsing the file
            {
                setThresholds( ( m_thresholds = m_parser->getThresholdList() ) );

                m_rejectFileStatus->set( WMEpochRejection::FILE_LOADED, true ); // show success notification
            }
            else
            {
                m_rejectFileStatus->set( WMEpochRejection::FILE_ERROR, true ); // show error notification
            }
        }

        // button applyBufferSize clicked
        if( m_applyBufferSize->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleApplyBufferSize();

            m_applyBufferSize->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        // button/trigger applyThresholds clicked
        if( m_applyThresholds->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleApplyThresholds();

            m_applyThresholds->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        // button/trigger moduleReset clicked
        if( m_resetModule->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleModuleReset();

            m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            m_moduleState.wait(); // wait for events like input-data or properties changed
        }

        // receive data form the input-connector
        emmIn.reset();
        if( !m_input->isEmpty() )
        {
            emmIn = m_input->getData();
        }
        const bool dataValid = ( emmIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the input-connector
        {
            // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
            debugLog() << "received data";

            process( emmIn );

            debugLog() << "finished rejection";
        }
    }
}

bool WMEpochRejection::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochRejection", "processCompute" );

    // show process visualization
    boost::shared_ptr< WProgress > rejectProcess = boost::shared_ptr< WProgress >( new WProgress( "Check data for rejection." ) );
    m_progress->addSubProgress( rejectProcess );

    // ---------- PROCESSING ----------
    viewUpdate( emmIn ); // update the GUI component

    if( checkBadChannels( emmIn ) )
    {
        // test the buffered epochs when new bad channels
        checkBufferedEpochs();
    }

    m_epochCount->set( m_epochCount->get() + 1, true ); // count number of received inputs

    if( m_rejection->doRejection( emmIn ) )
    {
        WBadEpochManager::instance()->getBuffer()->push_back( emmIn );

        m_badEpochCount->set( WBadEpochManager::instance()->getBuffer()->size(), true );
    }
    else
    {
        updateOutput( emmIn ); // no rejection: transfer to the ouput connector.
    }

    rejectProcess->finish(); // finish the process visualization

    return true;
}

bool WMEpochRejection::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}

bool WMEpochRejection::processReset( WLEMMCommand::SPtr labp )
{
    viewReset();

    m_input->clear();
    m_output->updateData( labp );

    m_epochCount->set( 0, true );
    m_epochCountValid->set( 0, true );
    m_epochCountInValid->set( 0, true );
    m_badChannelCount->set( 0, true );
    m_badEpochCount->set( 0, true );

    m_rejection->initRejection();

    WBadChannelManager::instance()->reset(); // reset channel buffer
    WBadEpochManager::instance()->reset(); // reset epoch buffer

    return true;
}

void WMEpochRejection::updateOutput( WLEMMeasurement::SPtr emm )
{
    updateBadChannels( emm ); // set bad channels in EMDs

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );

    m_output->updateData( cmd ); // update the output-connector after processing
    m_epochCountValid->set( m_epochCountValid->get() + 1, true ); // update valid epoch counter

    debugLog() << "output connector updated";
}

void WMEpochRejection::setThresholds( boost::shared_ptr< std::list< WThreshold > > thresholdList )
{
    if( m_modalityLabelMap == 0 )
    {
        return;
    }

    BOOST_FOREACH( WThreshold threshold, *thresholdList.get() )
    {
        if( m_modalityLabelMap->count( threshold.getModaliyType() ) )
        {
            m_modalityLabelMap->at( threshold.getModaliyType() )->set( threshold.getValue(), true );
        }
    }
}

void WMEpochRejection::updateBadChannels( WLEMMeasurement::SPtr emm )
{
    if( !WBadChannelManager::instance()->isMapEmpty() )
    {
        for( size_t i = 0; i < emm->getModalityCount(); ++i )
        {
            WLEModality::Enum modality = emm->getModality( i )->getModalityType();

            if( WBadChannelManager::instance()->hasBadChannels( modality ) )
            {
                emm->getModality( i )->setBadChannels( WBadChannelManager::instance()->getChannelList( modality ) );
            }
        }
    }
}

void WMEpochRejection::callbackRejectionTypeChanged()
{
    debugLog() << "callbackRejectionTypeChanged() called.";

    m_rejection =
                    m_rejectionTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WEpochRejection::SPtr > >()->getValue();
}

void WMEpochRejection::initThresholds()
{
    if( m_thresholds == 0 )
        m_thresholds.reset( new WThreshold::WThreshold_List() );

    m_thresholds->push_back( WThreshold( WLEModality::EEG, EEG_THRESHOLD ) );
    m_thresholds->push_back( WThreshold( WLEModality::EOG, EOG_THRESHOLD ) );
    m_thresholds->push_back( WThreshold( WLEModality::MEG_MAG, MEG_MAG_THRESHOLD ) );
    m_thresholds->push_back( WThreshold( WLEModality::MEG_GRAD, MEG_GRAD_THRESHOLD ) );

    setThresholds( m_thresholds );
    m_rejection->setThresholds( m_thresholds );
}

void WMEpochRejection::handleApplyBufferSize()
{
    WBadEpochManager::instance()->resizeBuffer( m_epochBufferSize->get() );
}

void WMEpochRejection::handleApplyThresholds()
{
    debugLog() << "Apply thresholds.";

    BOOST_FOREACH( WThreshold& value, *m_thresholds.get() )
    {
        if( m_modalityLabelMap->count( value.getModaliyType() ) )
        {
            value.setValue( m_modalityLabelMap->at( value.getModaliyType() )->get() );
        }
    }

    m_rejection->setThresholds( m_thresholds );

    handleModuleReset();

    checkBufferedEpochs(); // test all buffered epochs.
}

void WMEpochRejection::handleModuleReset()
{
    debugLog() << "Module reset.";

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( labp );
}

void WMEpochRejection::checkBufferedEpochs()
{
    WBadEpochManager::CircBuff::iterator it;
    for( it = WBadEpochManager::instance()->getBuffer()->begin(); it != WBadEpochManager::instance()->getBuffer()->end(); ++it )
    {
        if( !m_rejection->doRejection( *it ) )
        {
            WBadEpochManager::instance()->getBuffer()->erase( it );
        }
    }

    m_badEpochCount->set( WBadEpochManager::instance()->getBuffer()->size(), true );
}

bool WMEpochRejection::checkBadChannels( WLEMMeasurement::SPtr emm )
{
    bool rc = false;

    // TODO(maschke): when the EMMs bad channels differs from the current bad channels in the buffer manager,
    // merge the lists or set the EMM lists as new lists? Currently the channels will be merged.

    BOOST_FOREACH( WLEMData::SPtr modality, emm->getModalityList() )
    {
        if( !m_rejection->validModality( modality->getModalityType() ) )
        {
            continue;
        }

        if( modality->getBadChannels()->size() == 0 )
        {
            continue;
        }

        if( *WBadChannelManager::instance()->getChannelList( modality->getModalityType() ) != *modality->getBadChannels() )
        {
            WBadChannelManager::instance()->merge( modality->getModalityType(), modality->getBadChannels() );

            rc = true;
        }
    }

    return rc;
}

// file status messages
const std::string WMEpochRejection::NO_FILE_LOADED = "No file loaded.";
const std::string WMEpochRejection::LOADING_FILE = "Loading file ...";
const std::string WMEpochRejection::FILE_LOADED = "File successfully loaded.";
const std::string WMEpochRejection::FILE_ERROR = "Could not load file.";

// default thresholds
const double WMEpochRejection::EEG_THRESHOLD = 150e-6;
const double WMEpochRejection::EOG_THRESHOLD = 80e-6;
const double WMEpochRejection::MEG_MAG_THRESHOLD = 4e-12;
const double WMEpochRejection::MEG_GRAD_THRESHOLD = 200e-12;

// file labels for modalities
const std::string WMEpochRejection::EEG_LABEL = "eegReject";
const std::string WMEpochRejection::EOG_LABEL = "eogReject";
const std::string WMEpochRejection::MEG_GRAD_LABEL = "gradReject";
const std::string WMEpochRejection::MEG_MAG_LABEL = "magReject";
