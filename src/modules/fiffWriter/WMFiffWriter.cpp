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

#include <QFile>

#include <core/common/WPathHelper.h>

#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMFiffWriter.xpm"
#include "WMFiffWriter.h"

using namespace LaBP;
using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFiffWriter )

const std::string WMFiffWriter::ERROR = "error";
const std::string WMFiffWriter::COMPUTING = "computing";
const std::string WMFiffWriter::SUCCESS = "success";
const std::string WMFiffWriter::NONE = "none";
const std::string WMFiffWriter::FIFF_OK_TEXT = "FIFF ok";
const std::string WMFiffWriter::HD_LEADFIELD_OK_TEXT = "HD leadfield ok";
const std::string WMFiffWriter::READING = "reading ...";
const std::string WMFiffWriter::COMMAND = "leadfield";

WMFiffWriter::WMFiffWriter()
{
}

WMFiffWriter::~WMFiffWriter()
{
}

const std::string WMFiffWriter::getName() const
{
    return "FIFF Writer";
}

const std::string WMFiffWriter::getDescription() const
{
    // TODO(pieloth): module description
    return "TODO";
}

WModule::SPtr WMFiffWriter::factory() const
{
    return WModule::SPtr( new WMFiffWriter() );
}

const char** WMFiffWriter::getXPMIcon() const
{
    return module_xpm;
}

void WMFiffWriter::connectors()
{
    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in", "Expects a EMM-Command." ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "A loaded dataset." ) );
    addConnector( m_output );
}

void WMFiffWriter::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

//    m_fiffFile = m_properties->addProperty( "Sensor file:", "Read a FIFF file for sensor positions.", WPathHelper::getHomePath(),
//                    m_propCondition );
//    m_fiffFile->changed( true );
//
//    m_hdLeadfieldFile = m_properties->addProperty( "Leadfield file:", "Read a FIFF file for HD leadfield.",
//                    WPathHelper::getHomePath(), m_propCondition );
//    m_hdLeadfieldFile->changed( true );
//
//    m_start = m_properties->addProperty( "Interpolation:", "Start", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
//
//    m_status = m_properties->addProperty( "Status:", "Status", NONE );
//    m_status->setPurpose( PV_PURPOSE_INFORMATION );

}

void WMFiffWriter::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );
    m_moduleState.add( m_input->getDataChangedCondition() );

    ready();

    try
    {
        m_fiffWriter.reset( new WWriterFiff( "/tmp/test.fiff" ) );
    }
    catch( const WDHException& e )
    {
        errorLog() << "Could not create fiff writer!";
    }
    if( !m_fiffWriter->open() )
    {
        errorLog() << "Could not open writer!";
        m_fiffWriter.reset();
    }
//    if( !m_fiffWriter->close() )
//    {
//        errorLog() << "Could not close writer!";
//    }

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
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
    if( m_fiffWriter->close() )
    {
        infoLog() << "Close reader!";
    }
    else
    {
        errorLog() << "Could not close reader!";
    }
}

bool WMFiffWriter::processCompute( WLEMMeasurement::SPtr emm )
{
    bool rc = true;

    if( m_fiffWriter )
    {
        rc = m_fiffWriter->write( emm );
    }
//    if( m_leadfieldInterpolated )
//    {
//        // TODO NOTE: Manipulation of a incoming packet!!!
//        emm->getSubject()->setLeadfield( WEModalityType::EEG, m_leadfieldInterpolated );
//    }
//    else
//        if( m_fwdSolution )
//        {
//            m_fiffEmm = emm;
//            m_status->set( COMPUTING, true );
//            if( interpolate() )
//            {
//                emm->getSubject()->setLeadfield( WEModalityType::EEG, m_leadfieldInterpolated );
//                m_status->set( SUCCESS, true );
//            }
//            else
//            {
//                m_status->set( ERROR, true );
//            }
//        }
//        else
//        {
//            errorLog() << "No interpolated leadfield or no HD leadfield to compute!";
//            rc = false;
//        }

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return rc;
}

bool WMFiffWriter::processInit( WLEMMCommand::SPtr labp )
{
//    if( labp->hasEmm() )
//    {
//        m_fiffEmm = labp->getEmm();
//    }
//    if( m_fiffEmm && m_fwdSolution )
//    {
//        m_status->set( COMPUTING, true );
//        if( interpolate() )
//        {
//            WLEMMeasurement::SPtr emm = labp->getEmm();
//            emm->getSubject()->setLeadfield( WEModalityType::EEG, m_leadfieldInterpolated );
//            m_status->set( SUCCESS, true );
//        }
//        else
//        {
//            m_status->set( ERROR, true );
//        }
//    }

    m_output->updateData( labp );
    return true;
}

bool WMFiffWriter::processMisc( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMFiffWriter::processTime( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMFiffWriter::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}
