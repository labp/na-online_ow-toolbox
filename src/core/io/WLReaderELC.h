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

#ifndef WLREADERELC_H_
#define WLREADERELC_H_

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include "core/dataHandler/io/WReader.h"

#include "core/data/enum/WLEExponent.h"
#include "core/io/WLIOStatus.h"

class WLReaderELC: public WReader
{
public:
    static const std::string CLASS;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WLReaderELC( std::string fname );

    /**
     * Reads out a elc file. Positions are converted to millimeter, if necessary.
     */
    WLIOStatus::IOStatusT read( std::vector< WPosition >* const posOut, std::vector< std::string >* const labelsOut,
                    std::vector< WVector3i >* const facesOut );

private:
    WLIOStatus::IOStatusT readUnit( WLEExponent::Enum* const exp, const std::string& line );
    WLIOStatus::IOStatusT readNumPos( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readNumPoly( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readPositions( std::ifstream& ifs, size_t count, std::vector< WPosition >* const posOut );
    WLIOStatus::IOStatusT readLabels( std::ifstream& ifs, size_t count, std::vector< std::string >* const labelsOut );
    WLIOStatus::IOStatusT readPolygons( std::ifstream& ifs, size_t count, std::vector< WVector3i >* const facesOut );

    void convertToMilli( std::vector< WPosition >* const pos, WLEExponent::Enum exp );
};

#endif  // WLREADERELC_H_
