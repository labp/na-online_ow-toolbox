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

#include <vector>

#include "WLColorMap.h"
#include "WLColorMapClassic.h"

namespace LaBP
{

    WLColorMapClassic::WLColorMapClassic( ValueT min, ValueT max, WEColorMapMode::Enum mode ) :
                    WLColorMap( min, max, mode )
    {
        std::vector< ColorT > colors;
        colors.reserve( 3 );
        colors.push_back( osg::Vec4( 0.0, 0.0, 1.0, 1.0 ) );
        colors.push_back( osg::Vec4( 1.0, 1.0, 1.0, 1.0 ) );
        colors.push_back( osg::Vec4( 1.0, 0.0, 0.0, 1.0 ) );
        WLColorMap::setColors( colors );
    }

    WLColorMapClassic::~WLColorMapClassic()
    {
    }

    WEColorMap::Enum WLColorMapClassic::getType() const
    {
        return WEColorMap::CLASSIC;
    }

} /* namespace LaBP */
