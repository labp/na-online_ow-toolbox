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

#include "WLProfiler.h"

WLProfiler::WLProfiler( std::string source, std::string action, bool autoLog ) :
                m_source( source ), m_action( action ), m_autoLog( autoLog )
{
}

WLProfiler::~WLProfiler()
{
}

std::ostream& operator<<( std::ostream &strm, const WLProfiler& profiler )
{
    return profiler.write( strm );
}

std::string WLProfiler::getSource() const
{
    return m_source;
}

std::string WLProfiler::getAction() const
{
    return m_action;
}

bool WLProfiler::isAutoLog() const
{
    return m_autoLog;
}
