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

#include <vector>

#include "WLColorMap.h"
#include "WLColorMapHot.h"

WLColorMapHot::WLColorMapHot( ValueT min, ValueT max, WEColorMapMode::Enum mode ) :
                    WLColorMap( min, max, mode )
{
    std::vector< ColorT > colors;
    colors.reserve( 10 );
    colors.push_back( ColorT( 1.0, 1.0, 1.0, 1.0 ) );
    colors.push_back( ColorT( 1.0, 1.0, 0.75, 1.0 ) );
    colors.push_back( ColorT( 1.0, 1.0, 0.5, 1.0 ) );
    colors.push_back( ColorT( 1.0, 1.0, 0.25, 1.0 ) );
    colors.push_back( ColorT( 1.0, 1.0, 0.0, 1.0 ) );
    colors.push_back( ColorT( 1.0, 0.666, 0.0, 1.0 ) );
    colors.push_back( ColorT( 1.0, 0.333, 0.0, 1.0 ) );
    colors.push_back( ColorT( 1.0, 0.0, 0.0, 1.0 ) );
    colors.push_back( ColorT( 0.666, 0.0, 0.0, 1.0 ) );
    colors.push_back( ColorT( 0.333, 0.0, 0.0, 1.0 ) );

    WLColorMap::setColors( colors );
}

WLColorMapHot::~WLColorMapHot()
{
}

WEColorMap::Enum WLColorMapHot::getType() const
{
    return WEColorMap::HOT;
}
