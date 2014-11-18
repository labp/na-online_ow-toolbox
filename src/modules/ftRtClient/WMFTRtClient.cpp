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

#include <boost/exception/all.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/pointer_cast.hpp>

#include <core/common/WAssert.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMFTRtClient.h"
#include "WMFTRtClient.xpm"

// needed by the module loader.
W_LOADABLE_MODULE( WMFTRtClient )

static const std::string DEFAULT_FT_HOST = "localhost";
static const int DEFAULT_FT_PORT = 1972;

static const std::string CONNECTION_CONNECT = "Connect";
static const std::string CONNECTION_DISCONNECT = "Disconnect";

static const std::string CLIENT_STREAMING = "Streaming";
static const std::string CLIENT_NOT_STREAMING = "Not streaming";

WMFTRtClient::WMFTRtClient()
{
    m_stopStreaming = true;
}

WMFTRtClient::~WMFTRtClient()
{
}

WModule::SPtr WMFTRtClient::factory() const
{
    return WModule::SPtr( new WMFTRtClient() );
}

const char** WMFTRtClient::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMFTRtClient::getName() const
{
    return WLConstantsModule::generateModuleName( "FieldTrip Real-time Client" );
}

const std::string WMFTRtClient::getDescription() const
{
    return "Reads data for a FieldTrip Buffer and import them into Openwalnut. Module supports LaBP data types only!";
}

void WMFTRtClient::connectors()
{
    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMFTRtClient::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::hideComputeModalitySelection( true );

    //
    // init property container
    //
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    //
    // property group streaming client
    //
    m_propGrpFtClient = m_properties->addPropertyGroup( "FieldTrip Client", "FieldTrip Client", false );

    // connection type
    m_connectionType = WItemSelection::SPtr( new WItemSelection() );
    WItemSelectionItemTyped< CON_TYPE >::SPtr item;

    // TCP connection
    item.reset(
                    new WItemSelectionItemTyped< CON_TYPE >( CON_TCP, "TCP Connection",
                                    "Communicating with the FieldTrip buffer server using a TCP connection." ) );
    m_connectionType->addItem( item );

    // Unix connection
    item.reset(
                    new WItemSelectionItemTyped< CON_TYPE >( CON_UNIX, "Unix Connection",
                                    "Communicating with the FieldTrip buffer server using a Unix Domain Socket based connection." ) );
    m_connectionType->addItem( item );

    // getting the SelectorProperty from the list and add it to the properties
    m_connectionTypeSelection = m_propGrpFtClient->addProperty( "Connection Type:", "Choose a connection type.",
                    m_connectionType->getSelectorFirst(), boost::bind( &WMFTRtClient::cbConnectionTypeChanged, this ) );
    m_connectionTypeSelection->changed( true );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_connectionTypeSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_connectionTypeSelection );

    m_host = m_propGrpFtClient->addProperty( "Host IP:", "The hosts IP address providing the FieldTrip buffer.",
                    DEFAULT_FT_HOST );
    m_port = m_propGrpFtClient->addProperty( "Port number:",
                    "The port number on which the FieldTrip buffer server provides the data.", DEFAULT_FT_PORT );
    m_conStatus = m_propGrpFtClient->addProperty( "Connection status:",
                    "Shows the connections status to the FieldTrip buffer server.", CONNECTION_DISCONNECT );
    m_conStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgConnect = m_propGrpFtClient->addProperty( "Connect:", "Connect", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgDisconnect = m_propGrpFtClient->addProperty( "Disconnect:", "Disconnect", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgDisconnect->setHidden( true );

    m_blockSize = m_propGrpFtClient->addProperty( "Block size:", "Samples per packet.", 500 );

    m_streamStatus = m_propGrpFtClient->addProperty( "Streaming status:", "Shows the status of the streaming client.",
                    CLIENT_NOT_STREAMING );
    m_streamStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_applyScaling = m_propGrpFtClient->addProperty( "Apply scaling:", "Enable scale factor (range * cal).", false,
                    boost::bind( &WMFTRtClient::cbApplyScaling, this ) );
    m_trgStartStream = m_propGrpFtClient->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgStopStream = m_propGrpFtClient->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMFTRtClient::cbTrgStopStreaming, this ) );
    m_trgStopStream->setHidden( true );

    //
    // module reset button
    //
    m_resetModule = m_propGrpFtClient->addProperty( "Reset the module", "Reset", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_resetModule->changed( true );
}

void WMFTRtClient::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    m_connection.reset( new WFTConnection ); // create default connection

    m_ftRtClient.reset( new WFtbClient ); // create streaming client.

    cbConnectionTypeChanged();

    infoLog() << "Initializing module finished!";
}

void WMFTRtClient::moduleMain()
{
    moduleInit();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        if( m_trgConnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgConnect();
        }
        if( m_trgDisconnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgDisconnect();
        }
        if( m_trgStartStream->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgStartStreaming();
        }

        // button/trigger moduleReset clicked
        if( m_resetModule->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgReset();
            m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }
    }

    viewCleanup();
}

bool WMFTRtClient::processCompute( WLEMMeasurement::SPtr emmIn )
{
    viewUpdate( emmIn );
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emmIn );
    m_output->updateData( cmd );
    return true;
}

bool WMFTRtClient::processInit( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

bool WMFTRtClient::processReset( WLEMMCommand::SPtr cmd )
{
    viewReset();

    m_output->updateData( cmd );

    if( m_ftRtClient->isStreaming() )
    {
        hdlTrgDisconnect();
    }
    m_ftRtClient.reset( new WFtbClient );

    return true;
}

void WMFTRtClient::updateOutput( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );

    m_output->updateData( cmd ); // update the output-connector after processing
}

void WMFTRtClient::cbConnectionTypeChanged()
{
    debugLog() << __func__ << "() called.";

    const CON_TYPE con_type =
                    m_connectionTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< CON_TYPE > >()->getValue();
    switch( con_type )
    {
        case CON_TCP:
        {
            infoLog() << "Using TCP connection.";
            m_connection.reset( new WFTConnection );
            m_connection->setHost( m_host->get() );
            m_connection->setPort( m_port->get() );
            break;
        }
        case CON_UNIX:
        {
            infoLog() << "Using Unix connection.";
            m_connection.reset( new WFTConnection );
            const std::string pathname = m_host->get() + ":" + boost::lexical_cast< std::string >( m_port->get() );
            m_connection->setPath( pathname );
            break;
        }
        default:
            WAssert( false, "Unknown connection type!" );
    }
}

bool WMFTRtClient::hdlTrgConnect()
{
    debugLog() << __func__ << "() called.";

    cbConnectionTypeChanged(); // rebuild the connection

    infoLog() << "Establishing connection to FieldTrip Buffer Server: " << *m_connection;

    m_ftRtClient->setConnection( m_connection );

    if( m_ftRtClient->connect() )
    {
        infoLog() << "Connection to FieldTrip Buffer Server established successfully.";

        applyStatusConnected();

        return true;
    }
    else
    {
        errorLog() << "Connection to FieldTrip Buffer Server could not be established.";

        applyStatusDisconnected();

        return false;
    }
}

void WMFTRtClient::hdlTrgDisconnect()
{
    debugLog() << __func__ << "() called.";

    if( m_ftRtClient->isConnected() )
    {
        if( m_ftRtClient->isStreaming() )
        {
            cbTrgStopStreaming(); // stop streaming
        }

        m_ftRtClient->disconnect(); // disconnect client
    }

    applyStatusDisconnected();

    infoLog() << "Connection to FieldTrip Buffer Server closed.";
}

void WMFTRtClient::cbApplyScaling()
{
    if( m_ftRtClient )
    {
        m_ftRtClient->setApplyScaling( m_applyScaling->get( false ) );
    }
}

void WMFTRtClient::hdlTrgStartStreaming()
{
    debugLog() << __func__ << "() called.";

    if( !m_ftRtClient->isConnected() )
    {
        if( !hdlTrgConnect() )
        {
            applyStatusNotStreaming();
            return;
        }
    }

    // set some parameter on initializing the client
    m_ftRtClient->setBlockSize( m_blockSize->get() );
    cbApplyScaling();

    m_stopStreaming = false;

    WProgress::SPtr progress( new WProgress( "Start FieldTrip Streaming Client" ) );
    m_progress->addSubProgress( progress );

    if( m_ftRtClient->start() )
    {
        applyStatusStreaming(); // edit GUI for client status "streaming"

        m_progress->removeSubProgress( progress );

        debugLog() << "Header request on startup done. Beginning data streaming";

        while( !m_stopStreaming && !m_shutdownFlag() )
        {
            if( m_ftRtClient->fetchData() )
            {
                // get new samples
                WLEMMeasurement::SPtr emm( new WLEMMeasurement );
                if( m_ftRtClient->readEmm( emm ) )
                {
                    viewUpdate( emm ); // display on screen.
                    updateOutput( emm ); // transmit to the next module.
                    debugLog() << "Samples: " << emm->getModality(0)->getSamplesPerChan();

                }
                else
                {
                    errorLog() << "Error while reading data. The streaming will be stopped.";
                    m_stopStreaming = true;
                }
            }
            else
            {
                m_stopStreaming = true; // stop streaming on error during request.
                errorLog() << "Error while requesting buffer server for new data. Check your connection and the server, please.";
            }
        }
        m_ftRtClient->stop(); // stop streaming
    }
    else
    {
        m_progress->removeSubProgress( progress );
    }

    applyStatusNotStreaming();
}

void WMFTRtClient::cbTrgStopStreaming() // TODO(maschke): why called a second time on stopping?
{
    debugLog() << __func__ << "() called.";
    m_stopStreaming = true;
}

void WMFTRtClient::hdlTrgReset()
{
    debugLog() << __func__ << "() called.";

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( labp );
}

void WMFTRtClient::applyStatusConnected()
{
    m_trgConnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgDisconnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    m_trgConnect->setHidden( true );
    m_trgDisconnect->setHidden( false );
    m_conStatus->set( CONNECTION_CONNECT, true );
}

void WMFTRtClient::applyStatusDisconnected()
{
    m_trgConnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgDisconnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    m_trgConnect->setHidden( false );
    m_trgDisconnect->setHidden( true );
    m_conStatus->set( CONNECTION_DISCONNECT, true );
}

void WMFTRtClient::applyStatusStreaming()
{
    m_streamStatus->set( CLIENT_STREAMING, true );

    m_trgStartStream->setHidden( true );
    m_trgStopStream->setHidden( false );

    m_trgStartStream->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgStopStream->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMFTRtClient::applyStatusNotStreaming()
{
    m_streamStatus->set( CLIENT_NOT_STREAMING, true );

    m_trgStartStream->setHidden( false );
    m_trgStopStream->setHidden( true );

    m_ftRtClient->isConnected() ? applyStatusConnected() : applyStatusDisconnected();

    m_trgStartStream->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgStopStream->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}
