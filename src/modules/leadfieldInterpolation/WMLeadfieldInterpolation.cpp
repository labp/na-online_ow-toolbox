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

#include <string>

#include <mne/mne_forwardsolution.h>

#include <QFile>

#include <core/common/WPathHelper.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLMatrixTypes.h"
#include "core/io/WLReaderBND.h"
#include "core/io/WLReaderFIFF.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WLeadfieldInterpolation.h"
#include "WMLeadfieldInterpolation.xpm"
#include "WMLeadfieldInterpolation.h"

using namespace LaBP;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMLeadfieldInterpolation )

const std::string WMLeadfieldInterpolation::ERROR = "error";
const std::string WMLeadfieldInterpolation::COMPUTING = "computing";
const std::string WMLeadfieldInterpolation::SUCCESS = "success";
const std::string WMLeadfieldInterpolation::NONE = "none";
const std::string WMLeadfieldInterpolation::FIFF_OK_TEXT = "FIFF ok";
const std::string WMLeadfieldInterpolation::HD_LEADFIELD_OK_TEXT = "HD leadfield ok";
const std::string WMLeadfieldInterpolation::READING = "reading ...";

WMLeadfieldInterpolation::WMLeadfieldInterpolation()
{
}

WMLeadfieldInterpolation::~WMLeadfieldInterpolation()
{
}

const std::string WMLeadfieldInterpolation::getName() const
{
    return "Leadfield Interpolation";
}

const std::string WMLeadfieldInterpolation::getDescription() const
{
    // TODO(pieloth): module description
    return "TODO";
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
    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "A loaded dataset." ) );

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

void WMLeadfieldInterpolation::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );

    ready();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
        }

        if( ( m_start->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            m_status->set( COMPUTING, true );
            if( interpolate() )
            {
                m_status->set( SUCCESS, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
        }

        if( m_fiffFile->changed( true ) )
        {
            m_status->set( READING, true );
            if( readFiff( m_fiffFile->get().string() ) )
            {
                m_status->set( FIFF_OK_TEXT, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
        }

        if( m_hdLeadfieldFile->changed( true ) )
        {
            m_status->set( READING, true );
            if( readHDLeadfield( m_hdLeadfieldFile->get().string() ) )
            {
                m_status->set( HD_LEADFIELD_OK_TEXT, true );
            }
            else
            {
                m_status->set( ERROR, true );
            }
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
        if( fiffReader.Read( m_fiffEmm ) == WLReaderFIFF::ReturnCode::SUCCESS )
        {
            if( !m_fiffEmm->hasModality( WEModalityType::EEG ) )
            {
                errorLog() << "No EEG found!";
                // TODO(pieloth): Support for other modalities.
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
    // TODO(pieloth): Support for other modalities.
    debugLog() << "interpolate() called!";
    WLTimeProfiler tp( "WMLeadfieldInterpolation", "interpolate" );

    if( !m_fwdSolution || !m_fiffEmm )
    {
        errorLog() << "No FIFF or HDLeadfield file!";
        m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        return false;
    }

    if( !m_fiffEmm->hasModality( WEModalityType::EEG ) )
    {
        errorLog() << "No EEG available!";
        m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        return false;
    }

    WLeadfieldInterpolation li;
    li.prepareHDLeadfield( m_fwdSolution );
    li.setSensorPositions( m_fiffEmm->getModality( WEModalityType::EEG )->getAs< WLEMDEEG >()->getChannelPositions3d() );

    MatrixSPtr leadfield( new MatrixT( m_fiffEmm->getModality( WEModalityType::EEG )->getNrChans(), m_fwdSolution->nsource ) );
    bool success = li.interpolate( leadfield );
    if( success )
    {
        infoLog() << "Leadfield interpolation successful!";
    }
    else
    {
        errorLog() << "Could not interpolate leadfield!";
    }

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::MISC ) );
    // TODO(pieloth): Refactor "leadfield" to a global constant
    cmd->setMiscCommand( "leadfield" );
    WLEMMCommand::ParamT p = leadfield;
    cmd->setParameter( p );
    m_output->updateData( cmd );

    m_start->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    return success;
}
