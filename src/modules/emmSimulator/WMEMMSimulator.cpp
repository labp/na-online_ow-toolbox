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

#include <core/common/WRealtimeTimer.h>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WPacketizerEMM.h"
#include "WMEMMSimulator.h"
#include "WMEMMSimulator.xpm"

W_LOADABLE_MODULE( WMEMMSimulator )

using namespace LaBP;

std::string WMEMMSimulator::EStreaming::name( EStreaming::Enum val )
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
            return "Unkown state!";
    }
}

WMEMMSimulator::WMEMMSimulator()
{
    m_status = EStreaming::NO_DATA;
}

WMEMMSimulator::~WMEMMSimulator()
{
}

const std::string WMEMMSimulator::getName() const
{
    return "EMM Simulator";
}

const std::string WMEMMSimulator::getDescription() const
{
    return "Splits a EMM data in blocks and streams it through the signal processing chain.";
}

WModule::SPtr WMEMMSimulator::factory() const
{
    return WModule::SPtr( new WMEMMSimulator );
}

const char** WMEMMSimulator::getXPMIcon() const
{
    return module_xpm;
}

void WMEMMSimulator::connectors()
{
    WLModuleDrawable::connectors();

    m_input.reset( new WModuleInputData< WLEMMCommand >( shared_from_this(), "in", "Provides a filtered EMM-DataSet" ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "A loaded dataset." ) );

    // add it to the list of connectors. Please note, that a connector NOT added via addConnector will not work as expected.
    addConnector( m_output );
}

void WMEMMSimulator::properties()
{
    WLModuleDrawable::properties();
    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propAutoStart = m_properties->addProperty( "Auto start: ", "Start streaming when data is available.", true );

    m_propBlockSize = m_properties->addProperty( "Block size (ms): ", "Block size for streaming in milliseconds.", 1000 );
    m_propBlockSize->setMin( 1 );
    m_propBlockSize->setMax( 10000 );

    m_trgStart = m_properties->addProperty( "Start:", "Start streaming.", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_trgStop = m_properties->addProperty( "Stop:", "Stop streaming.", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMEMMSimulator::callbackStopTrg, this ) );

    m_propStatus = m_properties->addProperty( "Status:", "Status of streaming.", EStreaming::name( EStreaming::NO_DATA ) );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_propBlocksSent = m_properties->addProperty( "Blocks sent:", "Number of blocks which have been sent into processing chain.",
                    0 );
    m_propBlocksSent->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMEMMSimulator::moduleInit()
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

void WMEMMSimulator::moduleMain()
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
            handleStartTrg();
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
}

void WMEMMSimulator::reset()
{
    debugLog() << "reset() called!";
    EStreaming::Enum state = !m_data ? EStreaming::NO_DATA : EStreaming::READY;
    updateStatus( state );
    m_propBlockSize->set( 1000, true );
    m_propBlocksSent->set( 0, true );
}

void WMEMMSimulator::handleStartTrg()
{
    debugLog() << "handleStartTrg() called!";
    if( m_status == EStreaming::READY )
    {
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

void WMEMMSimulator::callbackStopTrg()
{
    debugLog() << "callbackStopTrg() called!";
    infoLog() << "Requesting streaming stop ...";
    updateStatus( EStreaming::STOP_REQUEST );
    m_trgStop->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMEMMSimulator::processCompute( WLEMMeasurement::SPtr emm )
{
    m_data = emm;
    updateStatus( EStreaming::READY );
    if(m_propAutoStart->get())
    {
        handleStartTrg();
    }
    return true;
}

bool WMEMMSimulator::processInit( WLEMMCommand::SPtr cmdIn )
{
    return processReset( cmdIn );
}

bool WMEMMSimulator::processReset( WLEMMCommand::SPtr cmdIn )
{
    reset();
    m_output->updateData( cmdIn );
    return true;
}

void WMEMMSimulator::stream()
{
    WPacketizerEMM packetizer( m_data, m_propBlockSize->get() );
    WRealtimeTimer waitTimer;
    const double SEC_PER_BLOCK = ( double )m_propBlockSize->get() / 1000; // blockSize in seconds

    WLEMMeasurement::SPtr emm;
    WLEMMCommand::SPtr cmd;
    size_t blocksSent = 0;
    while( m_status == EStreaming::STREAMING && packetizer.hasNext() && !m_shutdownFlag() )
    {
        // start
        waitTimer.reset();

        emm = packetizer.next();
        // Set a new profiler for the new EMM
        emm->setProfiler( WLLifetimeProfiler::instance( WLEMMeasurement::CLASS, "lifetime" ) );

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
