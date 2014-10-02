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

#ifndef WLREADERSOURCESPACE_H_
#define WLREADERSOURCESPACE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMSurface.h"
#include "core/io/WLReaderGeneric.h"

class WLReaderSourceSpace: public WLReaderGeneric< WLEMMSurface::SPtr >
{
public:
    static const std::string CLASS;

    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLReaderSourceSpace > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WLReaderSourceSpace > ConstSPtr;

    explicit WLReaderSourceSpace( std::string fname ) throw( WDHNoSuchFile );
    virtual ~WLReaderSourceSpace();

    virtual WLIOStatus::IOStatusT read( WLEMMSurface::SPtr* const surface );
};

#endif  // WLREADERSOURCESPACE_H_
