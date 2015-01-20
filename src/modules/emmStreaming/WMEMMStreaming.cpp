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
#include <core/common/WRealtimeTimer.h>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WMEMMStreaming.h"
#include "WMEMMSimulator.xpm"
#include "WPacketizerEMM.h"

W_LOADABLE_MODULE( WMEMMStreaming )

// -------------------------------------------------------------------------------------------------------------------------------
// Init
// -------------------------------------------------------------------------------------------------------------------------------

std::string WMEMMStreaming::EStreaming::name( EStreaming::Enum val )
{
    switch( val )
    {
        case NO_DATA:
            return "No data";
        case READY:
            return "Ready";
        case STREAMING:
            return "Streaming";
        case STOP_REQUEST:
            return "Requesting stop ...";
        default:
            return "Unknown state!";
    }
}

WMEMMStreaming::WMEMMStreaming()
{
    m_statusStreaming = EStreaming::NO_DATA;
}

WMEMMStreaming::~WMEMMStreaming()
{
}

// -------------------------------------------------------------------------------------------------------------------------------
// Module
// -------------------------------------------------------------------------------------------------------------------------------

const std::string WMEMMStreaming::getName() const
{
    return WLConstantsModule::generateModuleName("EMM Streaming");
}

const std::string WMEMMStreaming::getDescription() const
{
    return "Splits a EMM data in blocks and streams it through the signal processing chain.";
}

WModule::SPtr WMEMMStreaming::factory() const
{
    return WModule::SPtr( new WMEMMStreaming );
}

const char** WMEMMStreaming::getXPMIcon() const
{
    return module_xpm;
}

void WMEMMStreaming::connectors()
{
    WLModuleDrawable::connectors();

    m_input.reset(
                    new WModuleInputData< WLEMMCommand >( shared_from_this(), WLConstantsModule::CONNECTOR_NAME_IN,
                                    WLConstantsModule::CONNECTOR_DESCR_IN ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEMMStreaming::properties()
{
    WLModuleDrawable::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propAutoStart = m_properties->addProperty( "Auto start: ", "Start streaming when data is available.", true );

    m_propBlockSize = m_properties->addProperty( "Block size (ms): ", "Block size for streaming in milliseconds.", 1000 );
    m_propBlockSize->setMin( 1 );
    m_propBlockSize->setMax( 10000 );

    m_trgStart = m_properties->addProperty( "Start:", "Start streaming.", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_trgStop = m_properties->addProperty( "Stop:", "Stop streaming.", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMEMMStreaming::cbTrgStop, this ) );

    m_propStatusStreaming = m_properties->addProperty( "Status:", "Status of streaming.",
                    EStreaming::name( EStreaming::NO_DATA ) );
    m_propStatusStreaming->setPurpose( PV_PURPOSE_INFORMATION );

    m_propBlocksSent = m_properties->addProperty( "Blocks sent:", "Number of blocks which have been sent into processing chain.",
                    0 );
    m_propBlocksSent->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgReset = m_properties->addProperty( "Reset the module", "Reset", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgReset->changed( true );
}

void WMEMMStreaming::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );
}

void WMEMMStreaming::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();

        if( m_shutdownFlag() )
        {
            updateStatus( EStreaming::STOP_REQUEST );
            break;
        }

        if( ( m_trgStart->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            hdlTrgStart();
        }
        if( ( m_trgReset->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            hdlTrgReset();
        }

        bool dataUpdated = m_input->updated();
        cmdIn.reset();
        cmdIn = m_input->getData();
        bool dataValid = ( cmdIn );
        if( dataUpdated && dataValid )
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}

void WMEMMStreaming::reset()
{
    debugLog() << __func__ << "() called!";
    EStreaming::Enum state = !m_data ? EStreaming::NO_DATA : EStreaming::READY;
    updateStatus( state );
    m_propBlockSize->set( 1000, true );
    m_propBlocksSent->set( 0, true );
}

void WMEMMStreaming::hdlTrgReset()
{
    reset();
    m_trgReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

// -------------------------------------------------------------------------------------------------------------------------------
// Streaming
// -------------------------------------------------------------------------------------------------------------------------------

void WMEMMStreaming::hdlTrgStart()
{
    debugLog() << __func__ << "() called!";
    if( m_statusStreaming == EStreaming::READY )
    {
        debugLog() << "Sending reset command.";
        WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
        m_output->updateData( cmd );

        infoLog() << "Start streaming ...";
        updateStatus( EStreaming::STREAMING );
        stream();
        updateStatus( EStreaming::READY );
        infoLog() << "Finished streaming!";
    }
    else
    {
        warnLog() << "No data for streaming!";
    }
    m_trgStart->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMEMMStreaming::cbTrgStop()
{
    debugLog() << __func__ << "() called!";
    infoLog() << "Requesting streaming stop ...";
    updateStatus( EStreaming::STOP_REQUEST );
    m_trgStop->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMEMMStreaming::stream()
{
    const double SEC_PER_BLOCK = ( double )m_propBlockSize->get() / 1000.0; // blockSize in seconds
    WPacketizerEMM packetizer( m_data, SEC_PER_BLOCK * WLUnits::s );
    WRealtimeTimer waitTimer;


    WLEMMeasurement::SPtr emm;
    WLEMMCommand::SPtr cmd;
    size_t blocksSent = 0;
    while( m_statusStreaming == EStreaming::STREAMING && packetizer.hasNext() && !m_shutdownFlag() )
    {
        // start
        waitTimer.reset();

        emm = packetizer.next();
        // Set a new profiler for the new EMM
        emm->setProfiler( WLLifetimeProfiler::instance( WLEMMeasurement::CLASS, "lifetime" ) );

        if( blocksSent == 0 )
        {
            debugLog() << "Sending init command!";
            cmd = WLEMMCommand::instance( WLEMMCommand::Command::INIT );
            cmd->setEmm( emm );
            m_output->updateData( cmd );
        }

        cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
        cmd->setEmm( emm );
        viewUpdate( emm );
        m_output->updateData( cmd );
        m_propBlocksSent->set( ++blocksSent );

        // stop
        const double tuSleep = SEC_PER_BLOCK * 1000000 - waitTimer.elapsed() * 1000000;
        if( tuSleep > 0 )
        {
            boost::this_thread::sleep( boost::posix_time::microseconds( tuSleep ) );
            debugLog() << "Slept for " << tuSleep << " microseconds.";
        }
        else
        {
            warnLog() << "Streaming took " << abs( tuSleep ) << " microseconds to long!";
        }
    }
}

// -------------------------------------------------------------------------------------------------------------------------------
// Signal chain processing
// -------------------------------------------------------------------------------------------------------------------------------

bool WMEMMStreaming::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmdReset = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    m_output->updateData( cmdReset );

    m_data = emm;
    updateStatus( EStreaming::READY );

    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::INIT );
    cmd->setEmm( emm );
    m_output->updateData( cmd );

    if( m_propAutoStart->get() )
    {
        hdlTrgStart();
    }
    return true;
}

bool WMEMMStreaming::processInit( WLEMMCommand::SPtr cmdIn )
{
    return processReset( cmdIn );
}

bool WMEMMStreaming::processReset( WLEMMCommand::SPtr cmdIn )
{
    reset();
    m_output->updateData( cmdIn );
    return true;
}
