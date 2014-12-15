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

#ifndef WLREADERBND_H_
#define WLREADERBND_H_

#include <fstream>
#include <string>

#include "core/data/WLEMMBemBoundary.h"
#include "core/io/WLReaderGeneric.h"

/**
 * Reads BEM surface from BND file.
 *
 * \author pieloth
 * \ingroup io
 */
class WLReaderBND: public WLReaderGeneric< WLEMMBemBoundary::SPtr >
{
public:
    static const std::string CLASS;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WLReaderBND( std::string fname );

    virtual WLIOStatus::IOStatusT read( WLEMMBemBoundary::SPtr* const boundary );

private:
    WLIOStatus::IOStatusT readType( WLEMMBemBoundary::SPtr boundary, const std::string& line );
    WLIOStatus::IOStatusT readUnit( WLEMMBemBoundary::SPtr boundary, const std::string& line );
    WLIOStatus::IOStatusT readNumPos( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readNumPoly( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readPositions( std::ifstream& ifs, size_t count, WLEMMBemBoundary::SPtr boundary );
    WLIOStatus::IOStatusT readPolygons( std::ifstream& ifs, size_t count, WLEMMBemBoundary::SPtr boundary );
};

#endif  // WLREADERBND_H_
