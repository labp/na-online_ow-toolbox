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

#include <map>
#include <set>
#include <string>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderSourceSpace.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "reader/WLReaderDigPoints.h"
#include "reader/WLReaderBem.h"

#include "WMMneRtClient.h"
#include "WMMneRtClient.xpm"

using std::map;
using std::set;
using std::string;
using namespace LaBP;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMMneRtClient )

WMMneRtClient::WMMneRtClient() :
                m_stopStreaming( true )
{
}

WMMneRtClient::~WMMneRtClient()
{
    handleTrgConDisconnect();
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
    return "MNE Realtime Client";
}

const std::string WMMneRtClient::getDescription() const
{
    return "TODO";
}

void WMMneRtClient::connectors()
{
    // initialize connectors
    // TODO(pieloth) use OW class
    m_output.reset( new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "A loaded dataset." ) );

    // add it to the list of connectors. Please note, that a connector NOT added via addConnector will not work as expected.
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
    m_blockSize = m_propGrpConControl->addProperty( "Block size:", "Samples per packet.", block_size );

    // Setup streaming //
    m_propDataStatus = m_propGrpConControl->addProperty( "Data status:", "Streaming status.", STATUS_DATA_NOT_STREAMING );
    m_propDataStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgDataStart = m_propGrpConControl->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgDataStop = m_propGrpConControl->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMMneRtClient::callbackTrgDataStop, this ) );
    m_trgDataStop->setHidden( true );

    // Setup additional data //
    m_propGrpAdditional = m_properties->addPropertyGroup( "Additional data", "Additional data needed by other modules.", false );
    m_srcSpaceFile = m_propGrpAdditional->addProperty( "Source space file:", "Read a FIFF file containing the source space.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_srcSpaceFile->changed( true );

    m_bemFile = m_propGrpAdditional->addProperty( "BEM file:", "Read a FIFF file containing BEM layers.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_bemFile->changed( true );

    m_digPointsFile = m_propGrpAdditional->addProperty( "DigPoints file:", "Read a FIFF file containing digitization points.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_digPointsFile->changed( true );

    m_lfEEGFile = m_propGrpAdditional->addProperty( "Leadfield EEG file:", "Read a FIFF file containing the leadfield for EEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfEEGFile->changed( true );

    m_lfMEGFile = m_propGrpAdditional->addProperty( "Leadfield MEG file:", "Read a FIFF file containing the leadfield for MEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfMEGFile->changed( true );

    m_additionalStatus = m_propGrpAdditional->addProperty( "Additional data status:", "Additional data status.",
                    DATA_NOT_LOADED );
    m_additionalStatus->setPurpose( PV_PURPOSE_INFORMATION );
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
    m_subject.reset( new WLEMMSubject() );

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    if( m_srcSpaceFile->changed( true ) )
    {
        if( handleSurfaceFileChanged( m_srcSpaceFile->get().string() ) )
        {
            m_subject->setSurface( m_surface );
        }
    }
    if( m_bemFile->changed( true ) )
    {
        if( handleBemFileChanged( m_bemFile->get().string() ) )
        {
            m_subject->setBemBoundaries( m_bems );
        }
    }
    if( m_digPointsFile->changed( true ) )
    {
        if( handleDigPointsFileChanged( m_digPointsFile->get().string() ) )
        {
            // TODO(pieloth): set dig points
            m_rtClient->setDigPointsAndEEG( m_digPoints );
        }
    }
    if( m_lfEEGFile->changed( true ) )
    {
        if( handleLfFileChanged( m_lfEEGFile->get().string(), m_leadfieldEEG ) )
        {
            m_subject->setLeadfield( WEModalityType::EEG, m_leadfieldEEG );
        }
    }
    if( m_lfMEGFile->changed( true ) )
    {
        if( handleLfFileChanged( m_lfMEGFile->get().string(), m_leadfieldMEG ) )
        {
            m_subject->setLeadfield( WEModalityType::MEG, m_leadfieldMEG );
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
            handleTrgConConnect();
        }
        if( m_trgConDisconnect->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleTrgConDisconnect();
        }
        if( m_trgDataStart->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleTrgDataStart();
        }
        if( m_connectorSelection->changed( true ) )
        {
            handleTrgConnectorChanged();
        }

        if( m_srcSpaceFile->changed( true ) )
        {
            if( handleSurfaceFileChanged( m_srcSpaceFile->get().string() ) )
            {
                m_subject->setSurface( m_surface );
            }
        }
        if( m_bemFile->changed( true ) )
        {
            if( handleBemFileChanged( m_bemFile->get().string() ) )
            {
                m_subject->setBemBoundaries( m_bems );
            }
        }
        if( m_digPointsFile->changed( true ) )
        {
            if( handleDigPointsFileChanged( m_digPointsFile->get().string() ) )
            {
                m_rtClient->setDigPointsAndEEG( m_digPoints );
            }
        }
        if( m_lfEEGFile->changed( true ) )
        {
            if( handleLfFileChanged( m_lfEEGFile->get().string(), m_leadfieldEEG ) )
            {
                m_subject->setLeadfield( WEModalityType::EEG, m_leadfieldEEG );
            }
        }
        if( m_lfMEGFile->changed( true ) )
        {
            if( handleLfFileChanged( m_lfMEGFile->get().string(), m_leadfieldMEG ) )
            {
                m_subject->setLeadfield( WEModalityType::MEG, m_leadfieldMEG );
            }
        }
    }
}

void WMMneRtClient::handleTrgConConnect()
{
    debugLog() << "handleTrgConConnect() called!";

    m_rtClient.reset( new WRtClient( m_propConIp->get(), "OW-LaBP" ) );

    if( !m_digPoints.empty() )
    {
        m_rtClient->setDigPointsAndEEG( m_digPoints );
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

void WMMneRtClient::handleTrgConDisconnect()
{
    debugLog() << "handleTrgConDisconnect() called!";

    callbackTrgDataStop();
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
}

void WMMneRtClient::handleTrgDataStart()
{
    debugLog() << "handleTrgDataStart() called!";

    m_stopStreaming = false;

    m_rtClient->setSimulationFile( m_simFile->get() );
    m_rtClient->setBlockSize( m_blockSize->get() );
    viewReset();
    if( m_rtClient->start() )
    {
        m_propDataStatus->set( STATUS_DATA_STREAMING );
        m_trgDataStart->setHidden( true );
        m_trgDataStop->setHidden( false );

        while( !m_stopStreaming && !m_shutdownFlag() )
        {
            WLEMMeasurement::SPtr emm;
            if( m_rtClient->readData( emm ) )
            {
                if( m_subject && ( m_surface || m_bems || m_leadfieldEEG || m_leadfieldMEG ) )
                {
                    emm->setSubject( m_subject );
                }
                viewUpdate( emm );
                WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
                labp->setEmm( emm );
                m_output->updateData( labp );
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

void WMMneRtClient::callbackTrgDataStop()
{
    debugLog() << "callbackTrgDataStop() called!";
    m_stopStreaming = true;
}

void WMMneRtClient::handleTrgConnectorChanged()
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

bool WMMneRtClient::handleLfFileChanged( std::string fName, WLMatrix::SPtr& lf )
{
    debugLog() << "handleLfFileChanged()";
    m_additionalStatus->set( DATA_LOADING, true );

    WLReaderLeadfield::SPtr reader;
    try
    {
        reader.reset( new WLReaderLeadfield( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    if( reader->read( lf ) == WLIOStatus::SUCCESS )
    {
        m_additionalStatus->set( DATA_LOADED, true );
        return true;
    }
    else
    {
        m_additionalStatus->set( DATA_ERROR, true );
        errorLog() << "Could not read leadfield!";
        return false;
    }
}

bool WMMneRtClient::handleSurfaceFileChanged( std::string fName )
{
    debugLog() << "handleSurfaceFileChanged()";

    m_additionalStatus->set( DATA_LOADING, true );
    WLReaderSourceSpace::SPtr reader;
    try
    {
        reader.reset( new WLReaderSourceSpace( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    m_surface.reset( new WLEMMSurface() );
    if( reader->read( m_surface ) == WLIOStatus::SUCCESS )
    {
        m_additionalStatus->set( DATA_LOADED, true );
        return true;
    }
    else
    {
        m_additionalStatus->set( DATA_ERROR, true );
        errorLog() << "Could not read source space!";
        return false;
    }
}

bool WMMneRtClient::handleBemFileChanged( std::string fName )
{
    debugLog() << "handleBemFileChanged()";

    m_additionalStatus->set( DATA_LOADING, true );
    WLReaderBem::SPtr reader;
    try
    {
        reader.reset( new WLReaderBem( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    m_bems.reset( new std::vector< LaBP::WLEMMBemBoundary::SPtr > );
    if( reader->read( m_bems.get() ) )
    {
        m_additionalStatus->set( DATA_LOADED, true );
        infoLog() << "Loaded BEM layer: " << m_bems->size();
        return true;
    }
    else
    {
        m_additionalStatus->set( DATA_ERROR, true );
        errorLog() << "Could not read BEM layers!";
        return false;
    }
}

bool WMMneRtClient::handleDigPointsFileChanged( std::string fName )
{
    debugLog() << "handleDigPointsFileChanged()";

    m_additionalStatus->set( DATA_LOADING, true );
    WLReaderDigPoints::SPtr reader;
    try
    {
        reader.reset( new WLReaderDigPoints( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    if( reader->read( &m_digPoints ) == WLReaderDigPoints::ReturnCode::SUCCESS )
    {
        m_additionalStatus->set( DATA_LOADED, true );
        infoLog() << "Loaded dig points: " << m_digPoints.size();
        return true;
    }
    else
    {
        m_additionalStatus->set( DATA_ERROR, true );
        errorLog() << "Could not read dig points!";
        return false;
    }
}

bool WMMneRtClient::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    labp->setEmm( emm );
    m_output->updateData( labp );
    return true;
}

bool WMMneRtClient::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMMneRtClient::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}
