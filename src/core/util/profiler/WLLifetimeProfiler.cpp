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

#include <string>

#include "WLProfilerLogger.h"
#include "WLLifetimeProfiler.h"

const std::string WLLifetimeProfiler::CLASS = "WLLifetimeProfiler";

WLLifetimeProfiler::WLLifetimeProfiler( std::string source, std::string action ) :
                WLProfiler( source, action, false ), m_clones( 0 )
{
    m_timeProfiler.reset( new WLTimeProfiler( CLASS, "age", false ) );
    m_timeProfiler->start();
}

WLLifetimeProfiler::WLLifetimeProfiler( const WLLifetimeProfiler& o ) :
                WLProfiler( o.m_source, o.m_action, o.m_autoLog ), m_clones( o.m_clones + 1 )
{
    m_timeProfiler.reset( new WLTimeProfiler( *( o.m_timeProfiler ) ) );
    m_timeProfiler->start( false );
}

WLLifetimeProfiler::~WLLifetimeProfiler()
{
    if( isAutoLog() )
    {
        wlprofiler::log() << *this;
    }
}
