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

#include <QFile>

#include <core/common/WPathHelper.h>

#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMFiffWriter.xpm"
#include "WMFiffWriter.h"

using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFiffWriter )

static const std::string ERROR = "Error";
static const std::string OPEN = "Open";
static const std::string CLOSED = "Closed";
static const std::string NONE = "None";

WMFiffWriter::WMFiffWriter()
{
}

WMFiffWriter::~WMFiffWriter()
{
}

const std::string WMFiffWriter::getName() const
{
    return WLConstantsModule::generateModuleName( " FIFF Writer" );
}

const std::string WMFiffWriter::getDescription() const
{
    return "Creates and opens a FIFF file and writes incoming EMM data into it.";
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
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMFiffWriter::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propFile = m_properties->addProperty( "File:", "Destination file.", WPathHelper::getHomePath(), m_propCondition );
    m_propFile->changed( true );

    m_trgClose = m_properties->addProperty( "Close File:", "Close", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_propFileStatus = m_properties->addProperty( "Status:", "Status", NONE );
    m_propFileStatus->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMFiffWriter::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );
    m_moduleState.add( m_input->getDataChangedCondition() );

    ready();

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

        if( m_propFile->changed( true ) )
        {
            if( hdlFileChanged() )
            {
                m_propFileStatus->set( OPEN, true );
            }
            else
            {
                m_propFileStatus->set( ERROR, true );
            }
        }

        if( m_trgClose->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlFileClose();
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

    if( m_fiffWriter )
    {
        m_fiffWriter->close();
    }
}

bool WMFiffWriter::hdlFileChanged()
{
    if( m_fiffWriter )
    {
        m_fiffWriter->close();
        debugLog() << "Closed old writer!";
    }

    try
    {
        infoLog() << "Open file: " << m_propFile->get().string();
        m_fiffWriter.reset( new WWriterFiff( m_propFile->get().string() ) );
        return m_fiffWriter->open();
    }
    catch( const WDHException& e )
    {
        errorLog() << "Could not create fiff writer!";
        return false;
    }
}

void WMFiffWriter::hdlFileClose()
{
    if( m_fiffWriter )
    {
        m_fiffWriter->close();
        debugLog() << "Closed old writer!";
        m_fiffWriter.reset();
        m_propFileStatus->set( CLOSED );
    }

    m_trgClose->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMFiffWriter::processCompute( WLEMMeasurement::SPtr emm )
{
    bool rc = true;

    if( m_fiffWriter )
    {
        rc = m_fiffWriter->write( emm );
    }

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return rc;
}

bool WMFiffWriter::processInit( WLEMMCommand::SPtr labp )
{
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
    m_input->clear();
    hdlFileChanged();
    m_output->updateData( labp );
    return true;
}
