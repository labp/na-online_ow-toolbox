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
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderSourceSpace.h"
#include "core/io/WLReaderBem.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WPacketizerEMM.h"
#include "WMEMMSimulator.h"
#include "WMEMMSimulator.xpm"

W_LOADABLE_MODULE( WMEMMSimulator )

// -------------------------------------------------------------------------------------------------------------------------------
// Init
// -------------------------------------------------------------------------------------------------------------------------------

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
            return "Unknown state!";
    }
}

std::string WMEMMSimulator::EData::name( EData::Enum val )
{
    switch( val )
    {
        case DATA_NOT_LOADED:
            return "No data loaded.";
        case DATA_LOADING:
            return "Loading data ...";
        case DATA_LOADED:
            return "Data successfully loaded.";
        case DATA_ERROR:
            return "Could not load data.";
        default:
            return "Unknown state!";
    }
}

WMEMMSimulator::WMEMMSimulator()
{
    m_statusStreaming = EStreaming::NO_DATA;
}

WMEMMSimulator::~WMEMMSimulator()
{
}

// -------------------------------------------------------------------------------------------------------------------------------
// Module
// -------------------------------------------------------------------------------------------------------------------------------

const std::string WMEMMSimulator::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " EMM Simulator";
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

    m_input.reset(
                    new WModuleInputData< WLEMMCommand >( shared_from_this(), WLConstantsModule::CONNECTOR_NAME_IN,
                                    WLConstantsModule::CONNECTOR_DESCR_IN ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
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
                    boost::bind( &WMEMMSimulator::cbTrgStop, this ) );

    m_propStatusStreaming = m_properties->addProperty( "Status:", "Status of streaming.",
                    EStreaming::name( EStreaming::NO_DATA ) );
    m_propStatusStreaming->setPurpose( PV_PURPOSE_INFORMATION );

    m_propBlocksSent = m_properties->addProperty( "Blocks sent:", "Number of blocks which have been sent into processing chain.",
                    0 );
    m_propBlocksSent->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgReset = m_properties->addProperty( "Reset the module", "Reset", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgReset->changed( true );

    // Additional data
    // ---------------
    m_propGrpAdditional = m_properties->addPropertyGroup( "Additional Data", "Load additional data like BEM layer.", false );
    m_srcSpaceFile = m_propGrpAdditional->addProperty( "Source space file:", "Read a FIFF file containing the source space.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_srcSpaceFile->changed( true );

    m_bemFile = m_propGrpAdditional->addProperty( "BEM file:", "Read a FIFF file containing BEM layers.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_bemFile->changed( true );

    m_lfEEGFile = m_propGrpAdditional->addProperty( "Leadfield EEG file:", "Read a FIFF file containing the leadfield for EEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfEEGFile->changed( true );

    m_lfMEGFile = m_propGrpAdditional->addProperty( "Leadfield MEG file:", "Read a FIFF file containing the leadfield for MEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfMEGFile->changed( true );

    m_propStatusAdditional = m_propGrpAdditional->addProperty( "Additional data status:", "Additional data status.",
                    EData::name( EData::DATA_NOT_LOADED ) );
    m_propStatusAdditional->setPurpose( PV_PURPOSE_INFORMATION );
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
            hdlTrgStart();
        }
        if( ( m_trgReset->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            hdlTrgReset();
        }

        if( m_srcSpaceFile->changed( true ) )
        {
            hdlSurfaceFileChanged( m_srcSpaceFile->get().string() );
        }
        if( m_bemFile->changed( true ) )
        {
            hdlBemFileChanged( m_bemFile->get().string() );
        }

        if( m_lfEEGFile->changed( true ) )
        {
            hdlLeadfieldFileChanged( &m_leadfieldEEG, m_lfEEGFile->get().string() );
        }
        if( m_lfMEGFile->changed( true ) )
        {
            hdlLeadfieldFileChanged( &m_leadfieldMEG, m_lfMEGFile->get().string() );
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

void WMEMMSimulator::reset()
{
    debugLog() << "reset() called!";
    EStreaming::Enum state = !m_data ? EStreaming::NO_DATA : EStreaming::READY;
    updateStatus( state );
    m_propBlockSize->set( 1000, true );
    m_propBlocksSent->set( 0, true );

    m_subject.reset();

    m_srcSpaceFile->set( WPathHelper::getHomePath(), true );
    m_srcSpaceFile->changed( true );
    m_surface.reset();

    m_bemFile->set( WPathHelper::getHomePath(), true );
    m_bemFile->changed( true );
    if( ( m_bems ) )
    {
        m_bems->clear();
    }
    m_bems.reset();

    m_lfEEGFile->set( WPathHelper::getHomePath(), true );
    m_lfEEGFile->changed( true );
    m_leadfieldEEG.reset();

    m_lfMEGFile->set( WPathHelper::getHomePath(), true );
    m_lfMEGFile->changed( true );
    m_leadfieldMEG.reset();

    m_propStatusAdditional->set( EData::name( EData::DATA_NOT_LOADED ), true );
}

void WMEMMSimulator::hdlTrgReset()
{
    reset();
    m_trgReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

// -------------------------------------------------------------------------------------------------------------------------------
// Streaming
// -------------------------------------------------------------------------------------------------------------------------------

void WMEMMSimulator::hdlTrgStart()
{
    debugLog() << "handleStartTrg() called!";
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

void WMEMMSimulator::cbTrgStop()
{
    debugLog() << "callbackStopTrg() called!";
    infoLog() << "Requesting streaming stop ...";
    updateStatus( EStreaming::STOP_REQUEST );
    m_trgStop->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMEMMSimulator::stream()
{
    WPacketizerEMM packetizer( m_data, m_propBlockSize->get() );
    const bool set_subject = initAdditionalData( m_data->getSubject() );
    WRealtimeTimer waitTimer;
    const double SEC_PER_BLOCK = ( double )m_propBlockSize->get() / 1000; // blockSize in seconds

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
        if( set_subject )
        {
            emm->setSubject( m_subject );
        }

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

bool WMEMMSimulator::processCompute( WLEMMeasurement::SPtr emm )
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

// -------------------------------------------------------------------------------------------------------------------------------
// Additional data
// -------------------------------------------------------------------------------------------------------------------------------

bool WMEMMSimulator::hdlLeadfieldFileChanged( WLMatrix::SPtr* const lf, std::string fName )
{
    debugLog() << "handleLfFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading Leadfield" ) );
    m_progress->addSubProgress( progress );
    m_propStatusAdditional->set( EData::name( EData::DATA_LOADING ), true );

    WLReaderLeadfield::SPtr reader;
    try
    {
        reader.reset( new WLReaderLeadfield( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    if( reader->read( lf ) == WLIOStatus::SUCCESS )
    {
        m_propStatusAdditional->set( EData::name( EData::DATA_LOADED ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read leadfield!";
        m_propStatusAdditional->set( EData::name( EData::DATA_ERROR ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMEMMSimulator::hdlSurfaceFileChanged( std::string fName )
{
    debugLog() << "handleSurfaceFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading Surface" ) );
    m_progress->addSubProgress( progress );
    m_propStatusAdditional->set( EData::name( EData::DATA_LOADING ), true );

    WLReaderSourceSpace::SPtr reader;
    try
    {
        reader.reset( new WLReaderSourceSpace( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    m_surface.reset( new WLEMMSurface() );
    if( reader->read( &m_surface ) == WLIOStatus::SUCCESS )
    {
        m_propStatusAdditional->set( EData::name( EData::DATA_LOADED ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read source space!";
        m_propStatusAdditional->set( EData::name( EData::DATA_ERROR ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMEMMSimulator::hdlBemFileChanged( std::string fName )
{
    debugLog() << "handleBemFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading BEM Layer" ) );
    m_progress->addSubProgress( progress );
    m_propStatusAdditional->set( EData::name( EData::DATA_LOADING ), true );

    WLReaderBem::SPtr reader;
    try
    {
        reader.reset( new WLReaderBem( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    m_bems = WLList< WLEMMBemBoundary::SPtr >::instance();
    if( reader->read( m_bems.get() ) )
    {
        infoLog() << "Loaded BEM layer: " << m_bems->size();
        m_propStatusAdditional->set( EData::name( EData::DATA_LOADED ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read BEM layers!";
        m_propStatusAdditional->set( EData::name( EData::DATA_ERROR ), true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMEMMSimulator::initAdditionalData( WLEMMSubject::ConstSPtr subjectIn )
{
    if( !m_leadfieldEEG && !m_leadfieldMEG && !m_bems && !m_surface )
    {
        return false;
    }
    if( !m_subject )
    {
        m_subject = subjectIn->clone();
    }

    infoLog() << "Set additional data.";
    if( ( m_bems ) )
    {
        m_subject->setBemBoundaries( m_bems );
    }
    if( ( m_leadfieldEEG ) )
    {
        m_subject->setLeadfield( WLEModality::EEG, m_leadfieldEEG );
    }
    if( ( m_leadfieldMEG ) )
    {
        m_subject->setLeadfield( WLEModality::MEG, m_leadfieldMEG );
    }
    if( ( m_surface ) )
    {
        m_subject->setSurface( m_surface );
    }
    return true;
}
