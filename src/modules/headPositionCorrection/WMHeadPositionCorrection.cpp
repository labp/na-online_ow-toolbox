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

#include <string>

#include <core/common/WPathHelper.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMHpiInfo.h"
#include "core/data/emd/WLEMDHPI.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/io/WLReaderHpiInfo.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/WLDefines.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMHeadPositionCorrection.h"
#include "WMHeadPositionCorrection.xpm"

W_LOADABLE_MODULE( WMHeadPositionCorrection )

static const std::string STATUS_NO_REF_POS = "No reference position."; //!< No reference position available.
static const std::string STATUS_ERROR_READ = "Error on reading FIFF."; //!< Error on reading FIFF.
static const std::string STATUS_REF_POS = "Reference position loaded."; //!< Reference position available.

WMHeadPositionCorrection::WMHeadPositionCorrection() :
                m_hasRefPos( false )
{
}

WMHeadPositionCorrection::~WMHeadPositionCorrection()
{
}

WModule::SPtr WMHeadPositionCorrection::factory() const
{
    WMHeadPositionCorrection::SPtr instance( new WMHeadPositionCorrection() );
    return instance;
}

const std::string WMHeadPositionCorrection::getName() const
{
    return WLConstantsModule::generateModuleName( "Head Position Correction" );
}

const std::string WMHeadPositionCorrection::getDescription() const
{
    return "Corrects the head position, MEG only (in progress).";
}

const char** WMHeadPositionCorrection::getXPMIcon() const
{
    return module_xpm;
}

void WMHeadPositionCorrection::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMHeadPositionCorrection::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propGroup = m_properties->addPropertyGroup( "Head Position Correction", "Head Position Correction" );

    m_propMvThreshold = m_propGroup->addProperty( "Movement Threshold [m]:", "Movement Threshold for translation in meter.",
                    0.001 );
    m_propRadius = m_propGroup->addProperty( "Sphere Radius [m]:", "Sphere radius for dipole model in meter.", 0.07 );
    m_propPosFile = m_propGroup->addProperty( "Ref. Position:", "FIF file containing the reference position.",
                    WPathHelper::getHomePath(), m_propCondition );

    m_propStatus = m_propGroup->addProperty( "Status:", "Reports the status of actions.", STATUS_NO_REF_POS );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMHeadPositionCorrection::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // Wake up when input data changed
    m_moduleState.add( m_propCondition ); // Wake up when property changed

    ready(); // signal ready state
    waitRestored();

    m_hasRefPos = hdlPosFileChanged( m_propPosFile->get().string() );
    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    infoLog() << "Initializing module finished!";
}

void WMHeadPositionCorrection::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        if( m_propPosFile->changed( true ) )
        {
            m_hasRefPos = hdlPosFileChanged( m_propPosFile->get().string() );
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );
        if( dataValid )
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}

bool WMHeadPositionCorrection::processCompute( WLEMMeasurement::SPtr emm )
{
    if( !emm->hasModality( WLEModality::MEG ) )
    {
        errorLog() << "No MEG data available!";
        return false;
    }
    if( !emm->hasModality( WLEModality::HPI ) )
    {
        errorLog() << "No HPI data available!";
        return false;
    }
    if( !m_hasRefPos )
    {
        if( !emm->getHpiInfo()->getDevToHead()->empty() )
        {
            WLEMMHpiInfo::TransformationT::SPtr t = emm->getHpiInfo()->getDevToHead()->inverse();
            // TODO(pieloth): #393 Use WLTansformation.
            m_correction.setRefTransformation( t->data() );
            m_hasRefPos = true;
            m_propStatus->set( STATUS_REF_POS, true );
            infoLog() << "Set reference position from EMM: \n" << t;
        }
        else
        {
            errorLog() << "No reference head position!";
            return false;
        }
    }

    WLTimeProfiler profiler( getName(), __func__, true );

    WLEMDMEG::SPtr meg = emm->getModality< WLEMDMEG >( WLEModality::MEG );
    WLEMDHPI::SPtr hpi = emm->getModality< WLEMDHPI >( WLEModality::HPI );
    WLEMDMEG::SPtr megOut( new WLEMDMEG( *meg ) );

    if( !m_correction.isInitialzied() )
    {
        WLEMDMEG::createCoilInfos( meg.get() );
        m_correction.setMegCoilInfos( meg->getCoilInformation() );
        m_correction.setMovementThreshold( m_propMvThreshold->get( false ) );
        m_correction.setSphereRadius( m_propRadius->get( false ) );
        m_correction.init();
    }

    if( !m_correction.process( megOut.get(), *meg, *hpi ) )
    {
        errorLog() << "Error in correcting head position!";
        return false;
    }

    WLEMMeasurement::SPtr emmOut = emm->clone();
    std::vector< WLEMData::SPtr > mods = emm->getModalityList();
    for( size_t i = 0; i < mods.size(); ++i )
    {
        if( mods[i]->getModalityType() == WLEModality::MEG )
        {
            mods[i] = megOut;
            break;
        }
    }
    emmOut->setModalityList( mods );
    viewUpdate( emmOut );

    WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmdOut->setEmm( emmOut );
    m_output->updateData( cmdOut );

    return true;
}

bool WMHeadPositionCorrection::processInit( WLEMMCommand::SPtr cmdIn )
{
    infoLog() << "Initializing module.";
    m_correction.setMovementThreshold( m_propMvThreshold->get( false ) );
    m_correction.setSphereRadius( m_propRadius->get( false ) );

    if( cmdIn->hasEmm() )
    {
        WLEMMeasurement::SPtr emm = cmdIn->getEmm();
        if( emm->hasModality( WLEModality::MEG ) )
        {
            WLEMDMEG::SPtr meg = emm->getModality< WLEMDMEG >( WLEModality::MEG );
            WLEMDMEG::createCoilInfos( meg.get() );
            m_correction.setMegCoilInfos( meg->getCoilInformation() );
        }
        if( !m_hasRefPos && !emm->getHpiInfo()->getDevToHead()->empty() )
        {
            WLEMMHpiInfo::TransformationT::SPtr t = emm->getHpiInfo()->getDevToHead()->inverse();
            // TODO(pieloth): #393 Use WLTansformation.
            m_correction.setRefTransformation( t->data() );
            m_hasRefPos = true;
            m_propStatus->set( STATUS_REF_POS, true );
            infoLog() << "Set reference position from EMM: \n" << t;
        }
    }

    return m_correction.init();
}

bool WMHeadPositionCorrection::processReset( WLEMMCommand::SPtr cmdIn )
{
    WL_UNUSED( cmdIn );
    infoLog() << "Reseting module.";
    m_correction.reset();
    m_hasRefPos = hdlPosFileChanged( m_propPosFile->get().string() );
    return true;
}

bool WMHeadPositionCorrection::hdlPosFileChanged( std::string fName )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Reading ref. head position." ) );
    m_progress->addSubProgress( progress );

    WLEMMHpiInfo hpiInfo;
    bool rc = true;
    try
    {
        WLReaderHpiInfo reader( fName );
        if( reader.read( &hpiInfo ) != WLIOStatus::SUCCESS )
        {
            errorLog() << "Could not read reference head position!";
            m_propStatus->set( STATUS_ERROR_READ, true );
            rc = false;
        }
    }
    catch( const WException& e )
    {
        errorLog() << "Could not read reference head position!";
        m_propStatus->set( STATUS_ERROR_READ, true );
        rc = false;
    }

    if( rc )
    {
        WLEMMHpiInfo::TransformationT::SPtr t = hpiInfo.getDevToHead()->inverse();
        // TODO(pieloth): #393 Use WLTansformation.
        m_correction.setRefTransformation( t->data() );

        infoLog() << "Set reference position from FIFF:\n" << t;
        m_propStatus->set( STATUS_REF_POS, true );
    }
    progress->finish();
    m_progress->removeSubProgress( progress );
    return rc;
}
