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

#include <fstream>
#include <string>
#include <vector>

#include <core/common/WException.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include <Eigen/Core>

#include "WMCodeSnippets.h"
#include "WMCodeSnippets.xpm"

using namespace LaBP;
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
    return "Code Snippets";
}

const string WMCodeSnippets::getDescription() const
{
    return "Module to test small algorithms, import or export data and more.";
}

void WMCodeSnippets::connectors()
{
    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in", "Expects a EMM-Command." ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "A loaded dataset." ) );
    addConnector( m_output );
}

void WMCodeSnippets::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );
    m_writePos = m_properties->addProperty( "Write Positions:", "Writes positions from EEG/MEG to /tmp/", true, m_propCondition );
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
    }
}

bool WMCodeSnippets::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );

    if( m_writePos->get() )
    {
        writeEmdPositions( emm );
        m_writePos->set( false, true );
    }

    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return true;
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
    if( emm->hasModality( WEModalityType::EEG ) )
    {
        WLEMDEEG::ConstSPtr emd = emm->getModality< const WLEMDEEG >( WEModalityType::EEG );
        rc &= writeEmdPositions( emd->getChannelPositions3d().get(), "/tmp/positions_eeg.txt" );
    }
    if( emm->hasModality( WEModalityType::MEG ) )
    {
        WLEMDMEG::ConstSPtr emd = emm->getModality< const WLEMDMEG >( WEModalityType::MEG );
        rc &= writeEmdPositions( emd->getChannelPositions3d().get(), "/tmp/positions_meg.txt" );
    }

    WLEMMSubject::ConstSPtr subject = emm->getSubject();
    try
    {
        WLEMMSurface& surface = subject->getSurface( WLEMMSurface::Hemisphere::BOTH );
        rc &= writeEmdPositions( surface.getVertex().get(), "/tmp/positions_src.txt" );

        vector< WLEMMBemBoundary::SPtr >& bems = subject->getBemBoundaries();
        vector< WLEMMBemBoundary::SPtr >::const_iterator it = bems.begin();
        for(; it != bems.end(); ++it) {
            if((*it)->getBemType() == WEBemType::OUTER_SKIN) {
                vector< WPosition >& pos = (*it)->getVertex();
                rc &= writeEmdPositions( &pos, "/tmp/positions_skin.txt" );
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

bool WMCodeSnippets::writeEmdPositions( vector< WPosition >* const positions, string fname )
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
        if( fname.compare( "/tmp/positions_src.txt" ) == 0 || fname.compare( "/tmp/positions_skin.txt" ) == 0)
        {
            mat.setIdentity();
            mat *= 0.001;
        }
        else
        {
            mat.setIdentity();
        }

    vector< WPosition >::const_iterator it = positions->begin();
    for( ; it != positions->end(); ++it )
    {
        Eigen::Vector4f vec( it->x(), it->y(), it->z(), 1.0 );
        vec = mat * vec;
        fstream << vec.x() << "\t" << vec.y() << "\t" << vec.z() << std::endl;
    }
    fstream.close();
    return true;
}

