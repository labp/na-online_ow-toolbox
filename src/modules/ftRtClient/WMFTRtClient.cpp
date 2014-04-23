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

#include <boost/exception/all.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/pointer_cast.hpp>

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

#include "WFTNeuromagClient.h"
#include "fieldtrip/connection/WFTConnectionTCP.h"
#include "fieldtrip/connection/WFTConnectionUnix.h"
#include "fieldtrip/dataTypes/enum/WLEFTDataType.h"
#include "fieldtrip/dataTypes/WFTEventList.h"
#include "fieldtrip/io/request/WFTRequest_PutEvent.h"

#include "WMFTRtClient.h"

#include "WMFTRtClient.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFTRtClient )

WMFTRtClient::WMFTRtClient()
{
    m_stopStreaming = true;
}

WMFTRtClient::~WMFTRtClient()
{

}

boost::shared_ptr< WModule > WMFTRtClient::factory() const
{
    return boost::shared_ptr< WModule >( new WMFTRtClient() );
}

const char** WMFTRtClient::getXPMIcon() const
{
    return module_xpm;
}

/**
 * Returns the module name.
 */
const std::string WMFTRtClient::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " FieldTrip Real-time Client";
}

/**
 * Returns the module description.
 */
const std::string WMFTRtClient::getDescription() const
{
    return "Reads data for a FieldTrip Buffer and import them into Openwalnut. Module supports LaBP data types only!";
}

/**
 * Create the module connectors.
 */
void WMFTRtClient::connectors()
{
    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

/**
 * Define the property panel.
 */
void WMFTRtClient::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::hideComputeModalitySelection( true );

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    /*
     * property group streaming client
     */
    m_propGrpFtClient = m_properties->addPropertyGroup( "FieldTrip Client", "FieldTrip Client", false );

    // connection type
    m_connectionType = WItemSelection::SPtr( new WItemSelection() );
    boost::shared_ptr< WItemSelectionItemTyped< WFTConnection::SPtr > > item;
    WFTConnection::SPtr connection;

    // TCP connection
    connection.reset( new WFTConnectionTCP( DEFAULT_FT_HOST, DEFAULT_FT_PORT ) );
    item.reset(
                    new WItemSelectionItemTyped< WFTConnection::SPtr >( connection, "TCP Connection",
                                    "Communicating with the FieldTrip buffer server using a TCP connection." ) );
    m_connectionType->addItem( item );

    // Unix connection
    std::string unixPath = DEFAULT_FT_HOST + ":" + boost::lexical_cast< std::string >( DEFAULT_FT_PORT );
    connection.reset( new WFTConnectionUnix( unixPath ) );
    item.reset(
                    new WItemSelectionItemTyped< WFTConnection::SPtr >( connection, "Unix Connection",
                                    "Communicating with the FieldTrip buffer server using a Unix Domain Socket based connection." ) );
    m_connectionType->addItem( item );

    // getting the SelectorProperty from the list an add it to the properties
    m_connectionTypeSelection = m_propGrpFtClient->addProperty( "Connection Type:", "Choose a connection type.",
                    m_connectionType->getSelectorFirst(), boost::bind( &WMFTRtClient::callbackConnectionTypeChanged, this ) );
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
    m_waitTimeout = m_propGrpFtClient->addProperty( "Max. data request Timeout (ms):",
                    "Timeout at waiting for new data or events.", ( int )WFTRtClient::DEFAULT_WAIT_TIMEOUT );
    m_waitTimeout->setMin( 1 );
    m_waitTimeout->setMax( 100 );
    m_streamStatus = m_propGrpFtClient->addProperty( "Streaming status:", "Shows the status of the streaming client.",
                    CLIENT_NOT_STREAMING );
    m_streamStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgStartStream = m_propGrpFtClient->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgStopStream = m_propGrpFtClient->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMFTRtClient::callbackTrgStopStreaming, this ) );
    m_trgStopStream->setHidden( true );

    /* reset button */
    m_resetModule = m_propGrpFtClient->addProperty( "Reset the module", "Reset", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_resetModule->changed( true );

    /*
     * property group FieldTrip header
     */
    m_propGrpHeader = m_properties->addPropertyGroup( "FieldTrip Header information", "FieldTrip Header information", false );
    m_channels = m_propGrpHeader->addProperty( "Number of channels:", "Shows the number of channels.", 0 );
    m_channels->setPurpose( PV_PURPOSE_INFORMATION );
    m_frSample = m_propGrpHeader->addProperty( "Sampling frequency:", "Shows the sampling frequency.", 0.0 );
    m_frSample->setPurpose( PV_PURPOSE_INFORMATION );
    m_samples = m_propGrpHeader->addProperty( "Number of samples:", "Shows the number of samples read until now.", 0 );
    m_samples->setPurpose( PV_PURPOSE_INFORMATION );
    m_dataType = m_propGrpHeader->addProperty( "Data type:", "Data type", WLEFTDataType::name( WLEFTDataType::UNKNOWN ) );
    m_dataType->setPurpose( PV_PURPOSE_INFORMATION );
    m_events = m_propGrpHeader->addProperty( "Number of events:", "Shows the number of events read until now.", 0 );
    m_events->setPurpose( PV_PURPOSE_INFORMATION );
    m_headerBufSize = m_propGrpHeader->addProperty( "Additional header information (bytes):",
                    "Shows the number of bytes allocated by additional header information.", 0 );
    m_headerBufSize->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgShowChunks = m_propGrpHeader->addProperty( "Show Chunks", "Show Chunks", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );

    /*
     * property group FieldTrip buffer operations
     */
    m_propGrpBufferOperations = m_properties->addPropertyGroup( "FieldTrip Buffer Operations", "FieldTrip Buffer Operations",
                    false );
    m_trgFlushHeader = m_propGrpBufferOperations->addProperty( "Flush Header:", "Flush Header", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgFlushData = m_propGrpBufferOperations->addProperty( "Flush Data:", "Flush Data", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgFlushEvents = m_propGrpBufferOperations->addProperty( "Flush Events:", "Flush Events", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );

    m_trgPushEvent = m_propGrpBufferOperations->addProperty( "Push Event", "Push Event", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMFTRtClient::callbackTrgPushEvent, this ) );
}

/**
 * Method for initialize the module.
 */
void WMFTRtClient::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    m_connection.reset( new WFTConnectionTCP( DEFAULT_FT_HOST, DEFAULT_FT_PORT ) );

    m_ftRtClient.reset( new WFTNeuromagClient ); // create streaming client.

    callbackConnectionTypeChanged();

    infoLog() << "Initializing module finished!";
}

void WMFTRtClient::moduleMain()
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

        if( m_trgConnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgConnect();
        }
        if( m_trgDisconnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgDisconnect();
        }
        if( m_trgStartStream->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgStartStreaming();
        }

        // button/trigger moduleReset clicked
        if( m_resetModule->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgReset();

            m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        if( m_trgShowChunks->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgShowChunks();

            m_trgShowChunks->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        if( m_trgFlushHeader->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgFlushHeader();

            m_trgFlushHeader->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        if( m_trgFlushData->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgFlushData();

            m_trgFlushData->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        if( m_trgFlushEvents->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            callbackTrgFlushEvents();

            m_trgFlushEvents->set( WPVBaseTypes::PV_TRIGGER_READY, true );
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

            debugLog() << "finished";
        }
    }
}

bool WMFTRtClient::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMFTRtClient", "processCompute" );

    // show process visualization
    boost::shared_ptr< WProgress > process = boost::shared_ptr< WProgress >(
                    new WProgress( "Import data from FieldTrip Buffer." ) );
    m_progress->addSubProgress( process );

    // ---------- PROCESSING ----------
    viewUpdate( emmIn ); // update the GUI component

    updateOutput( emmIn );

    process->finish(); // finish the process visualization

    return true;
}

bool WMFTRtClient::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}

bool WMFTRtClient::processReset( WLEMMCommand::SPtr labp )
{
    viewReset();

    m_input->clear();
    m_output->updateData( labp );

    m_channels->set( 0, true );
    m_samples->set( 0, true );
    m_events->set( 0, true );
    m_frSample->set( 0.0, true );
    m_headerBufSize->set( 0, true );
    m_waitTimeout->set( ( int )WFTRtClient::DEFAULT_WAIT_TIMEOUT, true );
    m_dataType->set( WLEFTDataType::name( WLEFTDataType::UNKNOWN ), true );

    if( !m_ftRtClient->isStreaming() )
    {
        m_ftRtClient->resetClient();
    }

    return true;
}

void WMFTRtClient::updateOutput( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );

    m_output->updateData( cmd ); // update the output-connector after processing
}

void WMFTRtClient::callbackConnectionTypeChanged()
{
    debugLog() << "callbackConnectionTypeChanged() called.";

    m_connection =
                    m_connectionTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WFTConnection::SPtr > >()->getValue();

    if( typeid(WFTConnectionTCP) == typeid(*m_connection) )
    {
        boost::static_pointer_cast< WFTConnectionTCP >( m_connection )->set( m_host->get(), m_port->get() );
    }
    else
    {
        const std::string pathname = m_host->get() + ":" + boost::lexical_cast< std::string >( m_port->get() );
        boost::static_pointer_cast< WFTConnectionUnix >( m_connection )->set( pathname );
    }
}

bool WMFTRtClient::callbackTrgConnect()
{
    debugLog() << "callbackTrgConnect() called.";

    callbackConnectionTypeChanged(); // rebuild the connection

    infoLog() << "Establishing connection to FieldTrip Buffer Server with: " << m_connection->getName() << " ["
                    << m_connection->getConnectionString() << "].";

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

void WMFTRtClient::callbackTrgDisconnect()
{
    debugLog() << "callbackTrgDisconnect() called.";

    if( m_ftRtClient->isConnected() )
    {
        if( m_ftRtClient->isStreaming() )
        {
            callbackTrgStopStreaming(); // stop streaming
        }

        m_ftRtClient->disconnect(); // disconnect client

    }

    applyStatusDisconnected();

    infoLog() << "Connection to FieldTrip Buffer Server closed.";
}

void WMFTRtClient::callbackTrgStartStreaming()
{
    debugLog() << "callbackTrgStartStreaming() called.";

    if( !m_ftRtClient->isConnected() )
    {
        if( !callbackTrgConnect() )
        {
            applyStatusNotStreaming();
            return;
        }
    }

    // set some parameter on initializing the client
    m_ftRtClient->setTimeout( ( UINT32_T )m_waitTimeout->get() );

    m_stopStreaming = false;

    if( m_ftRtClient->start() )
    {
        applyStatusStreaming(); // edit GUI for client status "streaming"

        dispHeaderInfo(); // display header information
        //m_ftRtClient->printChunks(); // print the chunk buffers content.
        debugLog() << "Header request on startup done. Beginning data streaming";

        while( !m_stopStreaming && !m_shutdownFlag() )
        {
            if( m_ftRtClient->doWaitRequest( m_ftRtClient->getSampleCount(), m_ftRtClient->getEventCount() ) )
            {
                // get new samples
                if( m_ftRtClient->getNewSamples() )
                {
                    m_samples->set( m_ftRtClient->getSampleCount(), true );
                    m_channels->set( m_ftRtClient->getData()->getDataDef().nchans, true );
                    m_dataType->set(
                                    WLEFTDataType::name( ( WLEFTDataType::Enum )m_ftRtClient->getData()->getDataDef().data_type ),
                                    true );

                    WLEMMeasurement::SPtr emm( new WLEMMeasurement );

                    if( m_ftRtClient->createEMM( *emm ) )
                    {
                        viewUpdate( emm ); // display on screen.

                        updateOutput( emm ); // transmit to the next.
                    }
                    else
                    {
                        errorLog() << "Error while extracting values from response. The streaming will be stopped.";
                        m_stopStreaming = true;
                    }

                }

                // get new events
                if( m_ftRtClient->getNewEvents() )
                {
                    m_events->set( m_ftRtClient->getEventCount(), true );

                    BOOST_FOREACH(WFTEvent::SPtr event, *m_ftRtClient->getEventList())
                    {
                        debugLog() << "Fire Event: " << *event;
                    }
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

    applyStatusNotStreaming();
}

void WMFTRtClient::callbackTrgStopStreaming() // TODO(maschke): why called second times on stopping?
{
    debugLog() << "callbackTrgStopStreaming() called.";
    m_stopStreaming = true;
}

void WMFTRtClient::callbackTrgReset()
{
    debugLog() << "Module reset.";

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( labp );
}

void WMFTRtClient::callbackTrgShowChunks()
{
    debugLog() << "callbackTrgShowChunks() called.";

    m_ftRtClient->printChunks();
}

void WMFTRtClient::callbackTrgFlushHeader()
{
    debugLog() << "callbackTrgFlushHeader() called.";

    m_ftRtClient->doFlushHeaderRequest();
}

void WMFTRtClient::callbackTrgFlushData()
{
    debugLog() << "callbackTrgFlushData() called.";

    m_ftRtClient->doFlushDataRequest();
}

void WMFTRtClient::callbackTrgFlushEvents()
{
    debugLog() << "callbackTrgFlushEvents() called.";

    m_ftRtClient->doFlushEventsRequest();

}

void WMFTRtClient::callbackTrgPushEvent() // TODO(maschke): why called second times?
{
    debugLog() << "callbackTrgPushEvent() called.";

    m_trgPushEvent->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    std::string type = "Eventtyp";
    std::string value = "Hello World";

    WFTRequest::SPtr request( new WFTRequest_PutEvent( 10, 0, 0, type, value ) );
    WFTResponse::SPtr response( new WFTResponse );

    if( !m_ftRtClient->doRequest( *request, *response ) )
    {
        errorLog() << "Error while pushing event.";
        return;
    }

    if( response->checkPut() )
    {
        debugLog() << "Pushing event successful.";
    }
    else
    {
        errorLog() << "Failure on push event.";
    }

}

// TODO(maschke): Is it possible to enable/disable properties during runtime?
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

    m_channels->set( 0, true );
    m_samples->set( 0, true );
    m_frSample->set( 0.0, true );
    m_events->set( 0, true );
    m_headerBufSize->set( 0, true );
    m_dataType->set( WLEFTDataType::name( WLEFTDataType::UNKNOWN ), true );

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

void WMFTRtClient::dispHeaderInfo()
{
    m_channels->set( m_ftRtClient->getHeader()->getHeaderDef().nchans, true );
    m_samples->set( m_ftRtClient->getHeader()->getHeaderDef().nsamples, true );
    m_frSample->set( m_ftRtClient->getHeader()->getHeaderDef().fsample, true );
    m_events->set( m_ftRtClient->getHeader()->getHeaderDef().nevents, true );
    m_headerBufSize->set( m_ftRtClient->getHeader()->getHeaderDef().bufsize, true );
}

const std::string WMFTRtClient::DEFAULT_FT_HOST = "localhost";
const int WMFTRtClient::DEFAULT_FT_PORT = 1972;

const std::string WMFTRtClient::CONNECTION_CONNECT = "Connect";
const std::string WMFTRtClient::CONNECTION_DISCONNECT = "Disconnect";

const std::string WMFTRtClient::CLIENT_STREAMING = "Streaming";
const std::string WMFTRtClient::CLIENT_NOT_STREAMING = "Not streaming";
