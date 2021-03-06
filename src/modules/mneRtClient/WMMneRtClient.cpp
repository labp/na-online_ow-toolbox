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

#include <map>
#include <set>
#include <string>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/io/WLReaderBem.h"
#include "core/io/WLReaderIsotrak.h"
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderSourceSpace.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WMMneRtClient.h"
#include "WMMneRtClient.xpm"

using std::map;
using std::set;
using std::string;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMMneRtClient )

static const int NO_CONNECTOR = -1;

static const std::string STATUS_CON_CONNECTED = "Connected";
static const std::string STATUS_CON_DISCONNECTED = "Disconnected";
static const std::string STATUS_CON_ERROR = "Error";

static const std::string STATUS_DATA_STREAMING = "Streaming";
static const std::string STATUS_DATA_NOT_STREAMING = "Not streaming";

static const std::string DATA_NOT_LOADED = "No data loaded.";
static const std::string DATA_LOADING = "Loading data ...";
static const std::string DATA_LOADED = "Data successfully loaded.";
static const std::string DATA_ERROR = "Could not load data.";
static const std::string STANDARD_FILE_PATH = WPathHelper::getHomePath().string();

WMMneRtClient::WMMneRtClient() :
                m_stopStreaming( true )
{
}

WMMneRtClient::~WMMneRtClient()
{
    hdlTrgConDisconnect();
}

WModule::SPtr WMMneRtClient::factory() const
{
    return WModule::SPtr( new WMMneRtClient() );
}

const char** WMMneRtClient::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMMneRtClient::getName() const
{
    return WLConstantsModule::generateModuleName( "MNE Realtime Client" );
}

const std::string WMMneRtClient::getDescription() const
{
    return "Connection to a MNE Real-time Server.";
}

void WMMneRtClient::connectors()
{
    WLModuleDrawable::connectors();

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMMneRtClient::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition.reset( new WCondition() );

    // Setup connection control //
    m_propGrpConControl = m_properties->addPropertyGroup( "MNE Connection Control", "Connections settings for MNE server.",
                    false );

    //    const std::string con_ip_address = "127.0.0.1";
    const string con_ip_address = "192.168.100.1";
    m_propConIp = m_propGrpConControl->addProperty( "IP:", "IP Address of MNE server.", con_ip_address );
    m_propConStatus = m_propGrpConControl->addProperty( "Connection status:", "Shows connection status.",
                    STATUS_CON_DISCONNECTED );
    m_propConStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgConConnect = m_propGrpConControl->addProperty( "Connect:", "Connect", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgConDisconnect = m_propGrpConControl->addProperty( "Disconnect:", "Disconnect", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgConDisconnect->setHidden( true );

    m_connectorItem = WItemSelection::SPtr( new WItemSelection() );
    m_connectorItem->addItem(
                    WItemSelectionItemTyped< int >::SPtr( new WItemSelectionItemTyped< int >( NO_CONNECTOR, "none", "none" ) ) );
    m_connectorSelection = m_propGrpConControl->addProperty( "Connectors", "Choose a server connector.",
                    m_connectorItem->getSelectorFirst(), m_propCondition );
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_connectorSelection );

    const string sim_file = "/opt/naonline/emm_data/intershift/rawdir/is05a/is05a1.fif";
    m_simFile = m_propGrpConControl->addProperty( "Simulation File:", "Local path on server to simluation file.", sim_file );

    const int block_size = 500;
    m_blockSize = m_propGrpConControl->addProperty( "Block size [samples]:", "Samples per packet.", block_size );

    // TODO(pieloth): Is data stored without scaling by default? ... WMFiffWriter
    m_applyScaling = m_propGrpConControl->addProperty( "Apply scaling:", "Enable scale factor (range * cal).", false,
                    boost::bind( &WMMneRtClient::cbApplyScaling, this ) );

    // Setup streaming //
    m_propDataStatus = m_propGrpConControl->addProperty( "Data status:", "Streaming status.", STATUS_DATA_NOT_STREAMING );
    m_propDataStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgDataStart = m_propGrpConControl->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgDataStop = m_propGrpConControl->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMMneRtClient::cbTrgDataStop, this ) );
    m_trgDataStop->setHidden( true );

    // Setup additional data //
    m_propGrpAdditional = m_properties->addPropertyGroup( "Additional data", "Additional data needed by other modules.", false );

    m_digPointsFile = m_propGrpAdditional->addProperty( "DigPoints file:", "Read a FIFF file containing digitization points.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_digPointsFile->changed( true );

    m_additionalStatus = m_propGrpAdditional->addProperty( "Additional data status:", "Additional data status.",
                    DATA_NOT_LOADED );
    m_additionalStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgAdditionalReset = m_propGrpAdditional->addProperty( "Reset the additional information", "Reset",
                    WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgAdditionalReset->changed( true );
}

void WMMneRtClient::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );
    ready();
    waitRestored();

    const string ip = "127.0.0.1";
    const string alias = "OW-LaBP";
    m_rtClient.reset( new WRtClient( ip, alias ) );

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    if( m_digPointsFile->changed( true ) )
    {
        if( hdlDigPointsFileChanged( m_digPointsFile->get().string() ) )
        {
            // TODO(pieloth): set dig points
            m_rtClient->setDigPointsAndEEG( *m_digPoints.get() );
        }
    }

    infoLog() << "Restoring module finished!";
}

void WMMneRtClient::moduleMain()
{
    moduleInit();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
        }

        if( m_trgConConnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgConConnect();
        }
        if( m_trgConDisconnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgConDisconnect();
        }
        if( m_trgDataStart->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgDataStart();
        }
        if( m_connectorSelection->changed( true ) )
        {
            hdlTrgConnectorChanged();
        }

        if( m_digPointsFile->changed( true ) )
        {
            if( hdlDigPointsFileChanged( m_digPointsFile->get().string() ) )
            {
                m_rtClient->setDigPointsAndEEG( *m_digPoints.get() );
            }
        }
        if( m_trgAdditionalReset->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgAdditionalReset();

            m_trgAdditionalReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }
    }

    viewCleanup();
}

void WMMneRtClient::hdlTrgConConnect()
{
    debugLog() << __func__ << "() called!";

    m_rtClient.reset( new WRtClient( m_propConIp->get(), "OW-LaBP" ) );

    if( ( m_digPoints ) && !m_digPoints->empty() )
    {
        m_rtClient->setDigPointsAndEEG( *m_digPoints.get() );
    }

    m_rtClient->connect();
    if( m_rtClient->isConnected() )
    {
        map< int, string > cMap;
        const int selCon = m_rtClient->getConnectors( &cMap );
        map< int, string >::const_iterator itMap = cMap.begin();
        m_connectorItem->clear();
        for( ; itMap != cMap.end(); ++itMap )
        {
            m_connectorItem->addItem(
                            WItemSelectionItemTyped< int >::SPtr(
                                            new WItemSelectionItemTyped< int >( itMap->first, itMap->second, itMap->second ) ) );
        }
        m_connectorSelection->set( m_connectorItem->getSelector( selCon - 1 ) );

        m_trgConConnect->setHidden( true );
        m_trgConDisconnect->setHidden( false );
        m_propConStatus->set( STATUS_CON_CONNECTED );
    }
    else
    {
        m_trgConConnect->setHidden( false );
        m_trgConDisconnect->setHidden( true );
        m_propConStatus->set( STATUS_CON_ERROR );
    }
    m_trgConConnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgConDisconnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMMneRtClient::hdlTrgConDisconnect()
{
    debugLog() << __func__ << "() called!";

    cbTrgDataStop();
    m_rtClient->disconnect();

    if( !m_rtClient->isConnected() )
    {
        m_trgConConnect->setHidden( false );
        m_trgConDisconnect->setHidden( true );
        m_propConStatus->set( STATUS_CON_DISCONNECTED );
    }
    else
    {
        m_trgConConnect->setHidden( true );
        m_trgConDisconnect->setHidden( false );
        m_propConStatus->set( STATUS_CON_ERROR );
    }
    m_trgConConnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgConDisconnect->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    processReset( WLEMMCommand::instance( WLEMMCommand::Command::RESET ) );
}

void WMMneRtClient::hdlTrgDataStart()
{
    debugLog() << __func__ << "() called!";

    m_stopStreaming = false;

    m_rtClient->setSimulationFile( m_simFile->get() );
    m_rtClient->setBlockSize( m_blockSize->get() );
    cbApplyScaling();
    viewReset();
    if( m_rtClient->start() )
    {
        m_propDataStatus->set( STATUS_DATA_STREAMING );
        m_trgDataStart->setHidden( true );
        m_trgDataStop->setHidden( false );

        bool isFirst = true;
        while( !m_stopStreaming && !m_shutdownFlag() )
        {
            WLEMMeasurement::SPtr emm = m_rtClient->getEmmPrototype();
            if( m_rtClient->readEmm( emm ) )
            {
                if( isFirst )
                {
                    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::INIT ) );
                    cmd->setEmm( emm );
                    processInit( cmd );
                    isFirst = false;
                }

                processCompute( emm );
            }
        }
        m_rtClient->stop();
    }

    m_propDataStatus->set( STATUS_DATA_NOT_STREAMING );
    m_trgDataStart->setHidden( false );
    m_trgDataStop->setHidden( true );

    m_trgDataStart->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    m_trgDataStop->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMMneRtClient::cbTrgDataStop()
{
    debugLog() << __func__ << "() called!";
    m_stopStreaming = true;
}

void WMMneRtClient::cbApplyScaling()
{
    if( !m_rtClient )
    {
        return;
    }
    m_rtClient->setScaling( m_applyScaling->get( false ) );
}

void WMMneRtClient::hdlTrgConnectorChanged()
{
    debugLog() << "callbackTrgConnectorChanged() called!";

    const int conId = m_connectorSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< int > >()->getValue();
    if( conId == NO_CONNECTOR )
    {
        return;
    }
    if( m_rtClient->setConnector( conId ) )
    {
        infoLog() << "set connector: " << m_connectorSelection->get().at( 0 )->getName();
    }
}

bool WMMneRtClient::hdlDigPointsFileChanged( std::string fName )
{
    debugLog() << "handleDigPointsFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading Dig. Points" ) );
    m_progress->addSubProgress( progress );
    m_additionalStatus->set( DATA_LOADING, true );
    m_digPoints.reset( new WLList< WLDigPoint > );

    WLReaderIsotrak::SPtr reader;
    try
    {
        reader.reset( new WLReaderIsotrak( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    if( reader->read( m_digPoints.get() ) == WLIOStatus::SUCCESS )
    {
        infoLog() << "Loaded dig points: " << m_digPoints->size();
        m_additionalStatus->set( DATA_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read dig points!";
        m_additionalStatus->set( DATA_ERROR, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

void WMMneRtClient::hdlTrgAdditionalReset()
{
    debugLog() << "callbackTrgAdditionalReset()";

    m_additionalStatus->set( DATA_NOT_LOADED, true );

    m_digPointsFile->set( STANDARD_FILE_PATH, true );
    m_digPointsFile->changed( true );

    m_digPoints->clear();
    m_digPoints.reset();
}

inline bool WMMneRtClient::processCompute( WLEMMeasurement::SPtr emm )
{
    viewUpdate( emm );
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return true;
}

inline bool WMMneRtClient::processInit( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

inline bool WMMneRtClient::processReset( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}
