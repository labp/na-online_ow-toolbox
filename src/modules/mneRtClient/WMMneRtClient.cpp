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

#include <boost/filesystem.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/kernel/WModule.h>

// Output connector and data
// TODO(pieloth): use OW class
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/data/WLDataSetEMM.h"

#include "WMMneRtClient.h"
#include "WMMneRtClient.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMMneRtClient )

const int WMMneRtClient::NO_CONNECTOR = -1;

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
    m_output.reset(
                    new LaBP::WLModuleOutputDataCollectionable< LaBP::WLDataSetEMM >( shared_from_this(), "out",
                                    "A loaded dataset." ) );

    // add it to the list of connectors. Please note, that a connector NOT added via addConnector will not work as expected.
    addConnector( m_output );
}

void WMMneRtClient::properties()
{
    LaBP::WLModuleDrawable::properties();

    m_propCondition.reset( new WCondition() );

    // Experiment loader - Fiff properties //
    m_propGrpExperiment = m_properties->addPropertyGroup( "LaBP Experiment Loader", "LaBP Experiment Loader", false );

    m_fiffFileStatus = m_propGrpExperiment->addProperty( "FIFF file status:", "FIFF file status.", NO_FILE_LOADED );
    m_fiffFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_fiffFile = m_propGrpExperiment->addProperty( "FIFF file:", "Read a FIFF file for the data stream.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_fiffFile->changed( true );

    // Experiment loader - Metadata //
    m_expLoadStatus = m_propGrpExperiment->addProperty( "Metadata status:", "LaBP Experiment data status.", NO_DATA_LOADED );
    m_expLoadStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_expLoadTrigger = m_propGrpExperiment->addProperty( "Load data", "Load data", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_expLoadTrigger->setHidden( true );

    const std::string subject( "none" );
    m_expSubject = m_propGrpExperiment->addProperty( "Subject:", "Selected a subject.", subject );
    m_expSubject->setHidden( true );
    m_expSubject->setPurpose( PV_PURPOSE_INFORMATION );

    const std::string trial( "none" );
    m_expTrial = m_propGrpExperiment->addProperty( "Trial:", "Selected a trial.", trial );
    m_expTrial->setHidden( true );
    m_expTrial->setPurpose( PV_PURPOSE_INFORMATION );

    m_expBemFiles.reset( new WItemSelection() );

    m_expSurfaces.reset( new WItemSelection() );

    // Setup connection control //
    m_propGrpConControl = m_properties->addPropertyGroup( "MNE Connection Control", "Connections settings for MNE server.",
                    false );

//    const std::string con_ip_address = "127.0.0.1";
    const std::string con_ip_address = "141.57.41.164";
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

    const std::string sim_file = "/opt/naonline/emm_data/intershift/rawdir/is05a/is05a1.fif";
    m_simFile = m_propGrpConControl->addProperty( "Simulation File:", "Local path on server to simluation file.", sim_file );

    // Setup streaming //
    m_propDataStatus = m_propGrpConControl->addProperty( "Data status:", "Streaming status.", STATUS_DATA_NOT_STREAMING );
    m_propDataStatus->setPurpose( PV_PURPOSE_INFORMATION );
    m_trgDataStart = m_propGrpConControl->addProperty( "Start streaming:", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_trgDataStop = m_propGrpConControl->addProperty( "Stop streaming:", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMMneRtClient::callbackTrgDataStop, this ) );
    m_trgDataStop->setHidden( true );
}

void WMMneRtClient::initModule()
{
    infoLog() << "Initializing module ...";
    waitRestored();

    // TODO(pieloth)
    const std::string ip = "127.0.0.1";
    const std::string alias = "OW-LaBP";
    m_rtClient.reset( new WRtClient( ip, alias ) );
    initView( LaBP::WLEMDDrawable2D::WEGraphType::DYNAMIC );
    infoLog() << "Initializing module finished!";
}

void WMMneRtClient::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );

    ready();

    initModule();

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
        if( m_fiffFile->changed( true ) )
        {
            handleExtractExpLoader( m_fiffFile->get().string() );
        }
        if( m_expLoadTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleExperimentLoadChanged();
        }
        if( m_connectorSelection->changed( true ) )
        {
            handleTrgConnectorChanged();
        }
        // TODO(pieloth)
    }
}

void WMMneRtClient::handleTrgConConnect()
{
    debugLog() << "handleTrgConConnect() called!";

    m_rtClient.reset( new WRtClient( m_propConIp->get(), "OW-LaBP" ) );

    m_rtClient->connect();
    if( m_rtClient->isConnected() )
    {

        std::map< int, std::string > cMap;
        const int selCon = m_rtClient->getConnectors( &cMap );
        std::map< int, std::string >::const_iterator itMap = cMap.begin();
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
    if( m_rtClient->start() )
    {
        m_propDataStatus->set( STATUS_DATA_STREAMING );
        m_trgDataStart->setHidden( true );
        m_trgDataStop->setHidden( false );

        while( !m_stopStreaming && !m_shutdownFlag() )
        {
            LaBP::WLDataSetEMM::SPtr emm( new LaBP::WLDataSetEMM() );
            if( m_rtClient->readData( emm ) )
            {
                if( m_isExpLoaded )
                {
                    emm->setSubject( m_subject );
                }
                updateView( emm );
                m_output->updateData( emm );
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

void WMMneRtClient::handleExtractExpLoader( std::string fName )
{
    boost::filesystem::path fiffFile( fName );
    boost::filesystem::path expRoot = WLReaderExperiment::getExperimentRootFromFiff( fiffFile );
    const std::string subject = WLReaderExperiment::getSubjectFromFiff( fiffFile );
    const std::string trial = WLReaderExperiment::getTrialFromFiff( fiffFile );

    m_expSubject->set( subject );
    m_expSubject->setHidden( false );
    m_expTrial->set( trial );
    m_expTrial->setHidden( false );

    if( !boost::filesystem::is_directory( expRoot ) )
    {
        expRoot = expRoot.parent_path();
    }
    m_expReader.reset( new WLReaderExperiment( expRoot.string(), subject ) );

    std::set< std::string > bems = m_expReader->findBems();
    m_expBemFiles->clear();
    m_propGrpExperiment->removeProperty( m_expBemFilesSelection );
    if( !bems.empty() )
    {
        for( std::set< std::string >::iterator it = bems.begin(); it != bems.end(); ++it )
        {
            m_expBemFiles->addItem(
                            WItemSelectionItemTyped< string >::SPtr( new WItemSelectionItemTyped< string >( *it, *it ) ) );
        }
        m_expBemFilesSelection = m_propGrpExperiment->addProperty( "BEM Layers:", "Select BEM Layers to use.",
                        m_expBemFiles->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_expBemFilesSelection );
        m_expLoadTrigger->setHidden( false );
    }

    m_expSurfaces->clear();
    m_propGrpExperiment->removeProperty( m_expSurfacesSelection );
    std::set< std::string > surfaces = m_expReader->findSurfaceKinds();
    if( !surfaces.empty() )
    {
        for( std::set< std::string >::iterator it = surfaces.begin(); it != surfaces.end(); ++it )
        {
            m_expSurfaces->addItem(
                            WItemSelectionItemTyped< string >::SPtr( new WItemSelectionItemTyped< string >( *it, *it ) ) );
        }
        m_expSurfacesSelection = m_propGrpExperiment->addProperty( "Surfaces:", "Select surface to use.",
                        m_expSurfaces->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_expSurfacesSelection );
        m_expLoadTrigger->setHidden( false );
    }
}

void WMMneRtClient::handleExperimentLoadChanged()
{
    debugLog() << "handleExperimentLoadChanged() called!";

    bool rc = false;
    m_expLoadStatus->set( LOADING_DATA, true );

    m_subject.reset( new LaBP::WLEMMSubject() );

    const string bemFile = m_expBemFilesSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    rc |= m_expReader->readBem( bemFile, m_subject );

    const string surfaceType = m_expSurfacesSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    rc |= m_expReader->readSourceSpace( surfaceType, m_subject );

    const string trial = m_expTrial->get();
    rc |= m_expReader->readLeadFields( surfaceType, bemFile, trial, m_subject );

    if( rc )
    {
        m_expLoadStatus->set( DATA_LOADED, true );
        m_isExpLoaded = true;
    }
    else
    {
        m_expLoadStatus->set( DATA_ERROR, true );
        m_isExpLoaded = false;
    }

    m_expLoadTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );
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
