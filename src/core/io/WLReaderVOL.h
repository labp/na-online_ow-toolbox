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

#ifndef WLREADERVOL_H_
#define WLREADERVOL_H_

#include <fstream>
#include <list>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/io/WLReader.h"

namespace LaBP
{
    class WLReaderVOL: public WLReader
    {
    public:
        static const std::string CLASS;

        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WLReaderVOL( std::string fname );
        virtual ~WLReaderVOL()
        {
        }

        ReturnCode::Enum read( std::list< WLEMMBemBoundary::SPtr >* const boundaries );

    private:
        ReturnCode::Enum readNumBoundaries( std::string& line, size_t& count );
        ReturnCode::Enum readConductUnit( std::string& line, WLEUnit::Enum& unit );
        ReturnCode::Enum readConductivities( std::ifstream& ifs, std::list< WLEMMBemBoundary::SPtr >* const boundaries );
        ReturnCode::Enum readBndFiles( std::ifstream& ifs, std::string& line,
                        std::list< WLEMMBemBoundary::SPtr >* const boundaries );
    };
}
#endif  // WLREADERVOL_H_
