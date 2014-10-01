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

#ifndef WLCOLORMAPCLASSIC_H_
#define WLCOLORMAPCLASSIC_H_

#include <boost/shared_ptr.hpp>

#include "WLColorMap.h"

class WLColorMapClassic: public WLColorMap
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLColorMapClassic > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLColorMapClassic > ConstSPtr;

    WLColorMapClassic( ValueT min, ValueT max, WEColorMapMode::Enum mode );
    virtual ~WLColorMapClassic();

    virtual WEColorMap::Enum getType() const;
};

#endif  // WLCOLORMAPCLASSIC_H_
