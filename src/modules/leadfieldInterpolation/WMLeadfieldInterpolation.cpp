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
#include <vector>

#include <mne/mne_forwardsolution.h>

#include <QtCore/QFile>

#include <core/common/WPathHelper.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLDataTypes.h"
#include "core/io/WLReaderBND.h"
#include "core/io/WLReaderFIFF.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "core/util/WLGeometry.h"

#include "WLeadfieldInterpolation.h"
#include "WMLeadfieldInterpolation.xpm"
#include "WMLeadfieldInterpolation.h"

using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMLeadfieldInterpolation )

static const std::string ERROR = "error";
static const std::string COMPUTING = "computing";
static const std::string SUCCESS = "success";
static const std::string NONE = "none";
static const std::string FIFF_OK_TEXT = "FIFF ok";
static const std::string HD_LEADFIELD_OK_TEXT = "HD leadfield ok";
static const std::string READING = "reading ...";

WMLeadfieldInterpolation::WMLeadfieldInterpolation()
{
}

WMLeadfieldInterpolation::~WMLeadfieldInterpolation()
{
}

const std::string WMLeadfieldInterpolation::getName() const
{
    return WLConstantsModule::generateModuleName( "Leadfield Interpolation" );
}

const std::string WMLeadfieldInterpolation::getDescription() const
{
    return "Calculates an interpolated leadfield at each digitized electrode position by "
                    "averaging the leadfield columns of the electrode's nearest neighbors in the BEM layer.";
}

WModule::SPtr WMLeadfieldInterpolation::factory() const
{
    return WModule::SPtr( new WMLeadfieldInterpolation() );
}

const char** WMLeadfieldInterpolation::getXPMIcon() const
{
    return module_xpm;
}

void WMLeadfieldInterpolation::connectors()
{
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMLeadfieldInterpolation::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_fiffFile = m_properties->addProperty( "Sensor file:", "Read a FIFF file for sensor positions.", WPathHelper::getHomePath(),
                    m_propCondition );
    m_fiffFile->changed( true );

    m_hdLeadfieldFile = m_properties->addProperty( "Leadfield file:", "Read a FIFF file for HD leadfield.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_hdLeadfieldFile->changed( true );

    m_start = m_properties->addProperty( "Interpolation:", "Start", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_status = m_properties->addProperty( "Status:", "Status", NONE );
    m_status->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMLeadfieldInterpolation::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // Wake up when input data changed
    m_moduleState.add( m_propCondition ); // Wake up when input data changed

    ready(); // signal ready state
    waitRestored();

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    m_status->set( NONE, true );
    if( m_fiffFile->changed( true ) )
    {
        WProgress::SPtr progress( new WProgress( "Reading FIFF" ) );
        m_progress->addSubProgress( progress );
        m_status->set( READING, true );
        if( readFiff( m_fiffFile->get().string() ) )
        {
            m_status->set( FIFF_OK_TEXT, true );
        }
        else
        {
            m_status->set( ERROR, true );
        }
        progress->finish();
        m_progress->removeSubProgress( progress );
    }

    if( m_hdLeadfieldFile->changed( true ) )
    {
        WProgress::SPtr progress( new WProgress( "Reading HD Leadfield" ) );
        m_progress->addSubProgress( progress );
        m_status->set( READING, true );
        if( readHDLeadfield( m_hdLeadfieldFile->get().string() ) )
        {
            m_status->set( HD_LEADFIELD_OK_TEXT, true );
        }
        else
        {
            m_status->set( ERROR, true );
        }
        progress->finish();
        m_progress->removeSubProgress( progress );
    }

    infoLog() << "Restoring module finished!";
}

void WMLeadfieldInterpolation::moduleMain()
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
            break;
        }

        if( ( m_start->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            WProgress::SPtr progress( new WProgress( "Interpolating Leadfield" ) );
            m_progress->addSubProgress( progress );
            m_status->set( COMPUTING, true );
            if( interpolate() )
            {
                m_status->set( SUCCESS, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
            progress->finish();
            m_progress->removeSubProgress( progress );
        }

        if( m_fiffFile->changed( true ) )
        {
            WProgress::SPtr progress( new WProgress( "Reading FIFF" ) );
            m_progress->addSubProgress( progress );
            m_status->set( READING, true );
            if( readFiff( m_fiffFile->get().string() ) )
            {
                m_status->set( FIFF_OK_TEXT, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
            progress->finish();
            m_progress->removeSubProgress( progress );
        }

        if( m_hdLeadfieldFile->changed( true ) )
        {
            WProgress::SPtr progress( new WProgress( "Reading HD Leadfield" ) );
            m_progress->addSubProgress( progress );
            m_status->set( READING, true );
            if( readHDLeadfield( m_hdLeadfieldFile->get().string() ) )
            {
                m_status->set( HD_LEADFIELD_OK_TEXT, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
            progress->finish();
            m_progress->removeSubProgress( progress );
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( cmdIn );
        }
    }
}

bool WMLeadfieldInterpolation::readFiff( const std::string& fname )
{
    infoLog() << "Reading FIFF file: " << fname;
    if( boost::filesystem::exists( fname ) && boost::filesystem::is_regular_file( fname ) )
    {
        WLReaderFIFF fiffReader( fname );
        m_fiffEmm.reset( new WLEMMeasurement() );
        if( fiffReader.read( &m_fiffEmm ) == WLIOStatus::SUCCESS )
        {
            if( !m_fiffEmm->hasModality( WLEModality::EEG ) )
            {
                errorLog() << "No EEG found!";
                return false;
            }
            infoLog() << "Reading FIFF file finished!";
            return true;
        }
        else
        {
            errorLog() << "Could not read file! Maybe not in FIFF format.";
            return false;
        }
    }
    else
    {
        errorLog() << "File does not exist!";
        return false;
    }
}

bool WMLeadfieldInterpolation::readHDLeadfield( const std::string& fname )
{
    infoLog() << "Reading HD leadfield file: " << fname;
    if( boost::filesystem::exists( fname ) && boost::filesystem::is_regular_file( fname ) )
    {
        QFile fileIn( fname.c_str() );

        m_fwdSolution = MNELIB::MNEForwardSolution::SPtr( new MNELIB::MNEForwardSolution( fileIn ) );
        infoLog() << "Channel info: " << m_fwdSolution->info.chs.size();
        infoLog() << "Matrix size: " << m_fwdSolution->sol->data.rows() << "x" << m_fwdSolution->sol->data.cols();

        return true;
    }
    else
    {
        errorLog() << "File does not exist!";
        return false;
    }
}

bool WMLeadfieldInterpolation::interpolate()
{
    debugLog() << __func__ << "() called!";
    WLTimeProfiler tp( "WMLeadfieldInterpolation", __func__ );

    if( !m_fwdSolution || !m_fiffEmm )
    {
        errorLog() << "No FIFF or HDLeadfield file!";
        m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        return false;
    }

    if( !m_fiffEmm->hasModality( WLEModality::EEG ) )
    {
        errorLog() << "No EEG available!";
        m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        return false;
    }
    if( m_fiffEmm->getFidToACPCTransformation()->empty() )
    {
        errorLog() << "Error no transformation from HEAD to ACPC!";
        return false;
    }

    WLeadfieldInterpolation li;
    if( !li.prepareHDLeadfield( m_fwdSolution ) )
    {
        errorLog() << "Error during preparing HD leadfield!";
        return false;
    }
    const WLEMDEEG::PositionsT::ConstSPtr eeg_pos =
                    ( m_fiffEmm->getModality( WLEModality::EEG )->getAs< WLEMDEEG >()->getChannelPositions3d() );
    WLeadfieldInterpolation::PositionsT::SPtr eegPosTrans = *m_fiffEmm->getFidToACPCTransformation() * *eeg_pos;
    li.setSensorPositions( eegPosTrans );

    m_leadfieldInterpolated.reset(
                    new MatrixT( m_fiffEmm->getModality( WLEModality::EEG )->getNrChans(), m_fwdSolution->nsource ) );
    bool success = li.interpolate( m_leadfieldInterpolated );
    if( success )
    {
        infoLog() << "Leadfield interpolation successful!";
    }
    else
    {
        m_leadfieldInterpolated.reset();
        errorLog() << "Could not interpolate leadfield!";
    }

    m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    return success;
}

bool WMLeadfieldInterpolation::processCompute( WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WMLeadfieldInterpolation", __func__ );
    bool rc = true;
    if( m_leadfieldInterpolated )
    {
        emm->getSubject()->setLeadfield( WLEModality::EEG, m_leadfieldInterpolated );
    }
    else
        if( m_fwdSolution )
        {
            m_fiffEmm = emm;
            m_status->set( COMPUTING, true );
            if( interpolate() )
            {
                emm->getSubject()->setLeadfield( WLEModality::EEG, m_leadfieldInterpolated );
                m_status->set( SUCCESS, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
        }
        else
        {
            errorLog() << "No interpolated leadfield or no HD leadfield to compute!";
            rc = false;
        }

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return rc;
}

bool WMLeadfieldInterpolation::processInit( WLEMMCommand::SPtr cmdIn )
{
    WLTimeProfiler tp( "WMLeadfieldInterpolation", __func__ );
    bool rc = true;
    if( cmdIn->hasEmm() && m_fwdSolution )
    {
        m_fiffEmm = cmdIn->getEmm();

        m_status->set( COMPUTING, true );
        if( interpolate() )
        {
            WLEMMeasurement::SPtr emm = cmdIn->getEmm();
            emm->getSubject()->setLeadfield( WLEModality::EEG, m_leadfieldInterpolated );
            m_status->set( SUCCESS, true );
        }
        else
        {
            m_status->set( ERROR, true );
            rc = false;
        }
    }

    m_output->updateData( cmdIn );
    return rc;
}

bool WMLeadfieldInterpolation::processMisc( WLEMMCommand::SPtr cmdIn )
{
    m_output->updateData( cmdIn );
    return true;
}

bool WMLeadfieldInterpolation::processTime( WLEMMCommand::SPtr cmdIn )
{
    m_output->updateData( cmdIn );
    return true;
}

bool WMLeadfieldInterpolation::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_input->clear();
    m_output->updateData( cmdIn );
    return true;
}
