//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <map>
#include <string>
#include <typeinfo>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

// Input & output data
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMData.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/util/profiler/WLTimeProfiler.h"

#include "WMEpochRejection.h"
#include "WMEpochRejection.xpm"
#include "WBadEpochManager.h"
#include "WEpochRejection.h"
#include "WEpochRejectionSingle.h"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE (WMEpochRejection)

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

/**
 * Returns the module name.
 */
const std::string WMEpochRejection::getName() const
{
    return "Epoch Rejection";
}

/**
 * Returns the module description.
 */
const std::string WMEpochRejection::getDescription() const
{
    return "Checks the input values of each modality for defined thresholds. Module supports LaBP data types only!";
}

/**
 * Create the module connectors.
 */
void WMEpochRejection::connectors()
{
    m_input = LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

/**
 * Define the property panel.
 */
void WMEpochRejection::properties()
{
    LaBP::WLModuleDrawable::properties();
    LaBP::WLModuleDrawable::setTimerangeInformationOnly( true );

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    /* PropertyGroup: thresholds for the several modalities  */
    m_propGrpThresholds = m_properties->addPropertyGroup( "Rejection Thresholds", "Rejection Thresholds", false );
    m_propGrpEpoch = m_properties->addPropertyGroup( "Epoch Properties", "Epoch Properties", false );

    /* Rejection file properties */
    m_rejectFileStatus = m_propGrpThresholds->addProperty( "CFG file status:", "CFG file status.", NO_FILE_LOADED );
    m_rejectFileStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_rejectFile = m_propGrpThresholds->addProperty( "CFG file:", "Read a CFG file, which contains the thresholds.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_rejectFile->changed( true );

    /* Thresholds */
    m_eegThreshold = m_propGrpThresholds->addProperty( "EEG threshold", "Threshold for the EEG modality.", EEG_THRESHOLD );
    m_eogThreshold = m_propGrpThresholds->addProperty( "EOG threshold", "Threshold for the EOG modality.", EOG_THRESHOLD );
    m_megGradThreshold = m_propGrpThresholds->addProperty( "MEG gradiometer Threshold",
                    "Threshold for the MEG gradiometer channels.", MEG_GRAD_THRESHOLD );
    m_megMagThreshold = m_propGrpThresholds->addProperty( "MEG magnetometer Threshold",
                    "Threshold for the MEG magnetometer channels.", MEG_MAG_THRESHOLD );

    m_epochCount = m_propGrpEpoch->addProperty( "Epoch Count:", "Number of epochs.", 0 );
    m_epochCount->setPurpose( PV_PURPOSE_INFORMATION );

    m_epochCountValid = m_propGrpEpoch->addProperty( "Valid Epochs:", "Number of not rejected epochs.", 0 );
    m_epochCountValid->setPurpose( PV_PURPOSE_INFORMATION );

    m_epochCountInValid = m_propGrpEpoch->addProperty( "Invalid Epochs:", "Number of rejected epochs.", 0 );
    m_epochCountInValid->setPurpose( PV_PURPOSE_INFORMATION );
}

/**
 * Method for init the module.
 */
void WMEpochRejection::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_epochCount->set( 0, true );
    m_epochCountValid->set( 0, true );
    m_epochCountInValid->set( 0, true );

    waitRestored();

    initView( LaBP::WLEMDDrawable2D::WEGraphType::SINGLE );

    infoLog() << "Initializing module finished!";
}

void WMEpochRejection::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    WLEMMCommand::SPtr emmIn;
    //LaBP::WLTimeProfiler::SPtr profiler( new LaBP::WLTimeProfiler( getName(), "process" ) );
    //LaBP::WLTimeProfiler::SPtr profilerIn;

    m_rejecting.reset( new WEpochRejectionSingle() );
    m_parser.reset( new WThresholdParser() );

    ready(); // signal ready state

    moduleInit();

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        // parsing a ".cfg" file
        if( m_rejectFile->changed( true ) )
        {
            m_rejectFileStatus->set( WMEpochRejection::LOADING_FILE, true ); // change file notification

            if( m_parser->parse( m_rejectFile->get().string() ) ) // start parsing the file
            {
                assignNewValues( m_parser->getThresholds() ); // assign the parsed values to the members

                m_rejectFileStatus->set( WMEpochRejection::FILE_LOADED, true ); // show success notification
            }
            else
            {
                m_rejectFileStatus->set( WMEpochRejection::FILE_ERROR, true ); // show error notification
            }
        }

        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            m_moduleState.wait(); // wait for events like input-data or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        // receive data form he input-connector
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

/**
 * Method to assign the parsed value to the property members. The properties will be updated in the view.
 */
void WMEpochRejection::assignNewValues( std::map< std::string, double > valueList )
{
    // iterate all thresholds
    for( std::map< std::string, double >::iterator it = valueList.begin(); it != valueList.end(); it++ )
    {
        switch( hashit( ( *it ).first ) )
        {
            case eegReject:
                m_eegThreshold->set( ( *it ).second, true );
                break;
            case eogReject:
                m_eogThreshold->set( ( *it ).second, true );
                break;
            case gradReject:
                m_megGradThreshold->set( ( *it ).second, true );
                break;
            case magReject:
                m_megMagThreshold->set( ( *it ).second, true );
                break;
            default:
                break;
        }
    }
}

/**
 * Test a given String for a string pattern and return the equivalent enum object for better testing.
 */
WMEpochRejection::modality_code WMEpochRejection::hashit( std::string const& inString )
{
    if( inString == WMEpochRejection::EEG_LABEL )
        return eegReject;
    if( inString == WMEpochRejection::EOG_LABEL )
        return eogReject;
    if( inString == WMEpochRejection::MEG_GRAD_LABEL )
        return gradReject;
    if( inString == WMEpochRejection::MEG_MAG_LABEL )
        return magReject;

    return eNULL;
}

bool WMEpochRejection::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochRejection", "processCompute" );
    WLEMMeasurement::SPtr emmOut;
    bool rejected;

    // show process visualization
    boost::shared_ptr< WProgress > rejectProcess = boost::shared_ptr< WProgress >( new WProgress( "Check data for rejecting." ) );
    m_progress->addSubProgress( rejectProcess );

    // time profiler for the main loop
    /*
     profilerIn = emmIn->getTimeProfiler()->clone();
     profilerIn->stop();
     profiler->addChild( profilerIn );
     if( !profiler->isStarted() )
     {
     profiler->start();
     }
     */
    // TimeProfiler to measure the processing time
//            LaBP::WLTimeProfiler::SPtr rejProfiler = profiler->createAndAdd( WEpochRejection::CLASS, "rejecting" );
//            rejProfiler->start();
    // ---------- PROCESSING ----------
    m_rejecting->initRejection();
    m_rejecting->setThresholds( m_eegThreshold->get(), m_eogThreshold->get(), m_megGradThreshold->get(),
                    m_megMagThreshold->get() ); // serve parameter for processing

    rejected = m_rejecting->getRejection( emmIn ); // call the rejection process on the input

//            rejProfiler->stopAndLog(); // stop process profiler

    m_epochCount->set( m_epochCount->get() + 1, true ); // count number of received inputs

    rejectProcess->finish(); // finish the process visualization

    // transfer the output to the view
    updateView( emmIn ); // update the GUI component

    // deliver to output-connector if there was no failure
    if( rejected == false )
    {
        WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmd->setEmm( emmIn );

        m_output->updateData( cmd ); // update the output-connector after processing
        m_epochCountValid->set( m_epochCountValid->get() + 1, true );
        debugLog() << "output connector updated";
    }
    else
    {
        // In case of rejecting the epoch, the single channels of the epoch have to test.

        //WInvalidEpochManager::instance()->addEpoch( emmIn );

        m_epochCountInValid->set( m_epochCountInValid->get() + 1, true );
    }

    return true;
}

bool WMEpochRejection::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}

bool WMEpochRejection::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}

// file status messages
const std::string WMEpochRejection::NO_FILE_LOADED = "No file loaded.";
const std::string WMEpochRejection::LOADING_FILE = "Loading file ...";
const std::string WMEpochRejection::FILE_LOADED = "File successfully loaded.";
const std::string WMEpochRejection::FILE_ERROR = "Could not load file.";

// define standard values for the thresholds
const double WMEpochRejection::EEG_THRESHOLD = 150e-6;
const double WMEpochRejection::EOG_THRESHOLD = 80e-6;
const double WMEpochRejection::MEG_MAG_THRESHOLD = 4e-12;
const double WMEpochRejection::MEG_GRAD_THRESHOLD = 200e-12;

// file labels for modalities
const std::string WMEpochRejection::EEG_LABEL = "eegReject";
const std::string WMEpochRejection::EOG_LABEL = "eogReject";
const std::string WMEpochRejection::MEG_GRAD_LABEL = "gradReject";
const std::string WMEpochRejection::MEG_MAG_LABEL = "magReject";
