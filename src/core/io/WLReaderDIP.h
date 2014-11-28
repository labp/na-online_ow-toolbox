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

#ifndef WLREADERDIP_H_
#define WLREADERDIP_H_

#include <fstream>
#include <string>

#include "core/data/WLEMMSurface.h"

#include "core/io/WLReaderGeneric.h"

/**
 * Reads a surface from a DIP file.
 *
 * \author pieloth
 * \ingroup io
 */
class WLReaderDIP: public WLReaderGeneric< WLEMMSurface::SPtr >
{
public:
    static const std::string CLASS;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WLReaderDIP( std::string fname );

    virtual WLIOStatus::IOStatusT read( WLEMMSurface::SPtr* const surface );

private:
    WLIOStatus::IOStatusT readUnit( WLEMMSurface::SPtr surface, const std::string& line );
    WLIOStatus::IOStatusT readNumPos( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readNumPoly( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readPositions( std::ifstream& ifs, size_t count, WLEMMSurface::SPtr surface );
    WLIOStatus::IOStatusT readPolygons( std::ifstream& ifs, size_t count, WLEMMSurface::SPtr surface );
};

#endif  // WLREADERDIP_H_
