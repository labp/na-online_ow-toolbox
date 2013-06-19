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
#include <list>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/WLTimeProfiler.h"

#include "WMProfilerLog.h"
#include "WMProfilerLog.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMProfilerLog )

WMProfilerLog::WMProfilerLog()
{
}

WMProfilerLog::~WMProfilerLog()
{

}

boost::shared_ptr< WModule > WMProfilerLog::factory() const
{
    return boost::shared_ptr< WModule >( new WMProfilerLog() );
}

const char** WMProfilerLog::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMProfilerLog::getName() const
{
    return "Profile Logger";
}

const std::string WMProfilerLog::getDescription() const
{
    return "Profile Logger. Module supports LaBP data types only!";
}

void WMProfilerLog::connectors()
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

void WMProfilerLog::properties()
{
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    m_propGrpModule = m_properties->addPropertyGroup( "Profile Logger", "Profile Logger", false );

    const std::string file( "/tmp/ow_profiler.log" );
    m_file = m_propGrpModule->addProperty( "File:", "File incl. path to store log.", file );
}

void WMProfilerLog::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    WLEMMCommand::SPtr labpIn;

    ready(); // signal ready state

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        labpIn.reset();
        if( !m_input->isEmpty() )
        {
            labpIn = m_input->getData();
        }
        const bool dataValid = ( labpIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid && labpIn->hasEmm() ) // If there was an update on the inputconnector
        {
            // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
            debugLog() << "received data";
            write( m_file->get( true ), labpIn->getEmm() );
            m_output->updateData( labpIn );
        }
    }
}

bool WMProfilerLog::write( std::string fname, WLEMMeasurement::SPtr emm )
{
    std::ofstream fstream;
    fstream.open( fname.c_str(), std::ofstream::app );
    if( !fstream.is_open() )
        return false;

    WLTimeProfiler::SPtr profiler = emm->getTimeProfiler();
    profiler->stop();
    write( fstream, profiler, "" );

    fstream.close();
    return true;
}

void WMProfilerLog::write( std::ofstream& fstream, WLTimeProfiler::SPtr profiler, std::string prefix )
{
    fstream << prefix << profiler->getClass() << "::" << profiler->getAction() << ": " << profiler->getMilliseconds()
                    << std::endl;
    prefix.append( "\t" );
    // TODO(pieloth): Use new profiling structure.
//    std::list< WLTimeProfiler::SPtr >& profilers = profiler->getProfilers();
//    for( std::list< WLTimeProfiler::SPtr >::iterator it = profilers.begin(); it != profilers.end(); ++it )
//    {
//        write( fstream, ( *it ), prefix );
//    }
}
