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

#include <boost/lexical_cast.hpp>

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

#include "connection/WFTConnectionTCP.h"
#include "connection/WFTConnectionUnix.h"
#include "WFTRtClient.h"
#include "WFTClientStreaming.h"

#include "WMFTRtClient.h"

#include "WMFTRtClient.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFTRtClient )

WMFTRtClient::WMFTRtClient()
{

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
    m_doRequest = m_propGrpFtClient->addProperty( "Do FT Request", "do Request", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_doRequest->changed( true );

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

    WPropertyHelper::PC_SELECTONLYONE::addTo (m_connectionTypeSelection);
    WPropertyHelper::PC_NOTEMPTY::addTo( m_connectionTypeSelection );

    m_host = m_propGrpFtClient->addProperty( "Host IP:", "The hosts IP address providing the FieldTrip buffer.",
                    DEFAULT_FT_HOST );
    m_port = m_propGrpFtClient->addProperty( "Port number:",
                    "The port number on which the FieldTrip buffer server provides the data.", DEFAULT_FT_PORT );
    m_conStatus = m_propGrpFtClient->addProperty( "Connection status:",
                    "Shows the connections status to the FieldTrip buffer server.", CONNECTION_DISCONNECT );
    m_conStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgConnect = m_propGrpFtClient->addProperty( "Connect:", "Connect", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition,
                    boost::bind( &WMFTRtClient::callbackTrgConnect, this ) );
    m_trgDisconnect = m_propGrpFtClient->addProperty( "Disconnect:", "Disconnect", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition, boost::bind( &WMFTRtClient::callbackTrgDisconnect, this ) );
    m_trgDisconnect->setHidden( true );
    m_streamStatus = m_propGrpFtClient->addProperty( "Streaming status:", "Shows the status of the streaming client.",
                    CLIENT_NOT_STREAMING );
    m_streamStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgStartStream = m_propGrpFtClient->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition, boost::bind( &WMFTRtClient::callbackTrgStartStreaming, this ) );
    m_trgStopStream = m_propGrpFtClient->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition,
                    boost::bind( &WMFTRtClient::callbackTrgStopStreaming, this ) );
    m_trgStopStream->setHidden( true );

    /*
     * property group FieldTrip header
     */
    m_propGrpHeader = m_properties->addPropertyGroup( "FieldTrip Header information", "FieldTrip Header information", false );
    m_channels = m_propGrpFtClient->addProperty( "Number of channels:", "Shows the number of channels.", 0 );
    m_channels->setPurpose( PV_PURPOSE_INFORMATION );
    m_samples = m_propGrpFtClient->addProperty( "Number of samples:", "Shows the number of samples read until now.", 0 );
    m_samples->setPurpose( PV_PURPOSE_INFORMATION );
    m_frSample = m_propGrpFtClient->addProperty( "Sampling frequency:", "Shows the sampling frequency.", 0.0 );
    m_frSample->setPurpose( PV_PURPOSE_INFORMATION );
    m_events = m_propGrpFtClient->addProperty( "Number of events:", "Shows the number of events read until now.", 0 );
    m_events->setPurpose( PV_PURPOSE_INFORMATION );
    m_headerBufSize = m_propGrpFtClient->addProperty( "Additional header information (bytes):",
                    "Shows the number of bytes allocated by additional header information.", 0 );
    m_headerBufSize->setPurpose( PV_PURPOSE_INFORMATION );
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

    m_ftRtClient.reset( new WFTClientStreaming() );

    ready(); // signal ready state
    waitRestored();

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::SINGLE );

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

        // button applyBufferSize clicked
        if( m_doRequest->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleDoRequest();

            m_doRequest->set( WPVBaseTypes::PV_TRIGGER_READY, true );
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
    boost::shared_ptr< WProgress > rejectProcess = boost::shared_ptr< WProgress >(
                    new WProgress( "Import data from FieldTrip Buffer." ) );
    m_progress->addSubProgress( rejectProcess );

    // ---------- PROCESSING ----------
    viewUpdate( emmIn ); // update the GUI component

    rejectProcess->finish(); // finish the process visualization

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

    return true;
}

void WMFTRtClient::callbackConnectionTypeChanged()
{

}

void WMFTRtClient::callbackTrgConnect()
{

}

void WMFTRtClient::callbackTrgDisconnect()
{

}

void WMFTRtClient::callbackTrgStartStreaming()
{

}

void WMFTRtClient::callbackTrgStopStreaming()
{

}

void WMFTRtClient::handleDoRequest()
{
    WFTConnectionTCP::SPtr con( new WFTConnectionTCP( "localhost", 1972 ) );

    WFTRtClient::SPtr client( new WFTRtClient );
    client->setConnection( con );
    if( !client->connect() )
    {
        debugLog() << "Verbindungsaufbau fehltgeschlagen";
        return;
    }

    client->doReqest();
    return;
    if( con->isOpen() )
    {
        client->doReqest();
    }
    else
    {
        debugLog() << "keine offene Verbindung";
    }

    con->disconnect();
    debugLog() << "Verbindung geschlossen";
}

const std::string WMFTRtClient::DEFAULT_FT_HOST = "localhost";
const int WMFTRtClient::DEFAULT_FT_PORT = 1972;

const std::string WMFTRtClient::CONNECTION_CONNECT = "Connect";
const std::string WMFTRtClient::CONNECTION_DISCONNECT = "Disconnect";

const std::string WMFTRtClient::CLIENT_STREAMING = "Streaming";
const std::string WMFTRtClient::CLIENT_NOT_STREAMING = "Not streaming";
