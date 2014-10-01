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

#include <cmath> // M_PI, sin, pow
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include <core/common/WException.h>
#include <core/common/WRealtimeTimer.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/util/WLRingBuffer.h"

#include "WMCodeSnippets.h"
#include "WMCodeSnippets.xpm"

using std::ofstream;
using std::string;
using std::vector;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMCodeSnippets )

WMCodeSnippets::WMCodeSnippets()
{
}

WMCodeSnippets::~WMCodeSnippets()
{
}

const string WMCodeSnippets::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Code Snippets";
}

const string WMCodeSnippets::getDescription() const
{
    return "Module to test small algorithms, import or export data and more.";
}

void WMCodeSnippets::connectors()
{
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMCodeSnippets::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );
    m_writePos = m_properties->addProperty( "Write Positions:", "Writes positions from EEG/MEG to /tmp/", true, m_propCondition );

    m_trgGenerate = m_properties->addProperty( "Sinus generator:", "Start", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
}

WModule::SPtr WMCodeSnippets::factory() const
{
    return WModule::SPtr( new WMCodeSnippets() );
}

const char** WMCodeSnippets::getXPMIcon() const
{
    return module_xpm;
}

void WMCodeSnippets::moduleMain()
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

        if( ( m_trgGenerate->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            emulateSinusWave();
        }
    }
}

bool WMCodeSnippets::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );

    testExtract( emm );
//    if( m_writePos->get() )
//    {
//        writeEmdPositions( emm );
//        m_writePos->set( false, true );
//    }

    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return true;
}

void WMCodeSnippets::testExtract( WLEMMeasurement::SPtr emm )
{
    if( !emm->hasModality( WLEModality::MEG ) )
    {
        errorLog() << "No MEG.";
    }

    WLEMDMEG::SPtr meg = emm->getModality< WLEMDMEG >( WLEModality::MEG );
    infoLog() << *meg;

    WLEMDMEG::SPtr tmp;
    if( WLEMDMEG::extractCoilModality( tmp, meg, WLEModality::MEG ) )
        infoLog() << *tmp;
    else
        errorLog() << "Could not extract!";
    if( WLEMDMEG::extractCoilModality( tmp, meg, WLEModality::MEG_MAG ) )
        infoLog() << *tmp;
    else
        errorLog() << "Could not extract!";
    if( WLEMDMEG::extractCoilModality( tmp, meg, WLEModality::MEG_GRAD ) )
        infoLog() << *tmp;
    else
        errorLog() << "Could not extract!";
    if( WLEMDMEG::extractCoilModality( tmp, meg, WLEModality::MEG_GRAD_MERGED ) )
        infoLog() << *tmp;
    else
        errorLog() << "Could not extract!";
}

bool WMCodeSnippets::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMCodeSnippets::processMisc( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMCodeSnippets::processTime( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMCodeSnippets::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMCodeSnippets::writeEmdPositions( WLEMMeasurement::ConstSPtr emm )
{
    bool rc = true;
    if( emm->hasModality( WLEModality::EEG ) )
    {
        WLEMDEEG::ConstSPtr emd = emm->getModality< const WLEMDEEG >( WLEModality::EEG );
        rc &= writeEmdPositions( *emd->getChannelPositions3d(), "/tmp/positions_eeg.txt" );
    }
    if( emm->hasModality( WLEModality::MEG ) )
    {
        WLEMDMEG::ConstSPtr emd = emm->getModality< const WLEMDMEG >( WLEModality::MEG );
        rc &= writeEmdPositions( *emd->getChannelPositions3d(), "/tmp/positions_meg.txt" );
    }

    WLEMMSubject::ConstSPtr subject = emm->getSubject();
    try
    {
        WLEMMSurface::ConstSPtr surface = subject->getSurface( WLEMMSurface::Hemisphere::BOTH );
        rc &= writeEmdPositions( *surface->getVertex(), "/tmp/positions_src.txt" );

        const std::list< WLEMMBemBoundary::SPtr >& bems = *subject->getBemBoundaries();
        std::list< WLEMMBemBoundary::SPtr >::const_iterator it;
        for( it = bems.begin(); it != bems.end(); ++it )
        {
            if( ( *it )->getBemType() == WLEBemType::OUTER_SKIN || ( *it )->getBemType() == WLEBemType::HEAD )
            {
                const vector< WPosition >& pos = *( *it )->getVertex();
                rc &= writeEmdPositions( pos, "/tmp/positions_skin.txt" );
                break;
            }
        }

    }
    catch( const WException& e )
    {
        errorLog() << "No source space both available!";
    }
    return rc;
}

bool WMCodeSnippets::writeEmdPositions( const vector< WPosition >& positions, string fname )
{
    ofstream fstream;
    fstream.open( fname.c_str(), ofstream::out );
    if( !fstream.is_open() )
    {
        return false;
    }

    Eigen::Matrix4f mat;
    if( fname.compare( "/tmp/positions_meg.txt" ) == 0 )
    {
        debugLog() << "transforming: " << fname;
        mat << 0.99732f, 0.0693495f, -0.0232939f, -0.00432357f, -0.0672088f, 0.994307f, 0.0826812f, -0.00446779f, 0.0288952f, -0.0808941f, 0.996304f, 0.0442954f, 0.0f, 0.0f, 0.0f, 1.0f;
    }
    else
        if( fname.compare( "/tmp/positions_src.txt" ) == 0 || fname.compare( "/tmp/positions_skin.txt" ) == 0 )
        {
            mat.setIdentity();
            mat *= 0.001;
        }
        else
        {
            mat.setIdentity();
        }

    vector< WPosition >::const_iterator it = positions.begin();
    for( ; it != positions.end(); ++it )
    {
        Eigen::Vector4f vec( it->x(), it->y(), it->z(), 1.0 );
        vec = mat * vec;
        fstream << vec.x() << "\t" << vec.y() << "\t" << vec.z() << std::endl;
    }
    fstream.close();
    return true;
}

void WMCodeSnippets::generateSinusWave( WLEMData::DataT* const in, float sr, float f, float amp, float offset )
{
    float delta = 1 / static_cast< float >( sr );
    for( WLEMData::DataT::Index row = 0; row < in->rows(); ++row )
    {
        for( WLEMData::ChannelT::Index col = 0; col < in->cols(); ++col )
        {
            const WLEMData::ScalarT x = col * delta;
            const WLEMData::ScalarT x_rad = 2 * M_PI * x;
            const WLEMData::ScalarT y = amp * sin( f * x_rad ) + offset;
            ( *in )( row, col ) = y;
        }
    }
}

void WMCodeSnippets::emulateSinusWave()
{
    const float length = 30.0;
    const float sampling_frequency = 1000.0;
    const float sinus_frequency = 100.0;
    const size_t block_size = 1000;
    const float block_length = block_size / sampling_frequency;
    const size_t channels = 5;
    const float amp_factor = 10;
    const float offset = 5;

    WLEMMeasurement::SPtr emmPrototype( new WLEMMeasurement() );
    WLEMDEEG::SPtr eegPrototype( new WLEMDEEG() );
    eegPrototype->setSampFreq( sampling_frequency );

    WRealtimeTimer waitTimer;
    float seconds = 0.0;
    while( seconds < length )
    {
        waitTimer.reset();
        seconds += block_length;

        WLEMData::DataSPtr data( new WLEMData::DataT( channels, block_size ) );
        generateSinusWave( data.get(), sampling_frequency, sinus_frequency, amp_factor, offset );

        WLEMData::SPtr emd = eegPrototype->clone();
        emd->setData( data );
        WLEMMeasurement::SPtr emm = emmPrototype->clone();
        emm->addModality( emd );

        WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
        cmd->setEmm( emm );
        m_output->updateData( cmd );

        const double tuSleep = block_length * 1000000 - ( waitTimer.elapsed() * 1000000 );
        if( tuSleep > 0 )
        {
            boost::this_thread::sleep( boost::posix_time::microseconds( tuSleep ) );
            debugLog() << "Slept for " << tuSleep << " microseconds.";
        }
        else
        {
            warnLog() << "Generation took " << abs( tuSleep ) << " microseconds to long!";
        }
    }

    m_trgGenerate->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

