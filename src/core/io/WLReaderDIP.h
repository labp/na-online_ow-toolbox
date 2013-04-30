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

#ifndef WLREADERDIP_H_
#define WLREADERDIP_H_

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMSurface.h"

#include "core/io/WLReader.h"

namespace LaBP
{

    class WLReaderDIP: public WLReader
    {
    public:
        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WLReaderDIP( std::string fname );

        ReturnCode::Enum read( boost::shared_ptr< WLEMMSurface > surface );

    private:
        ReturnCode::Enum readUnit( std::string& line, boost::shared_ptr< WLEMMSurface > surface );
        ReturnCode::Enum readNumPos( std::string& line, size_t& count );
        ReturnCode::Enum readNumPoly( std::string& line, size_t& count );
        ReturnCode::Enum readPositions( std::ifstream& ifs, size_t count, boost::shared_ptr< WLEMMSurface > surface );
        ReturnCode::Enum readPolygons( std::ifstream& ifs, size_t count, boost::shared_ptr< WLEMMSurface > surface );
    };
}
#endif /* WLREADERDIP_H_ */
