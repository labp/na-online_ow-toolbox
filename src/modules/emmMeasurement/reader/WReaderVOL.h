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

#ifndef WREADERVOL_H_
#define WREADERVOL_H_

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/common/math/linearAlgebra/WVectorFixed.h"

#include "core/dataHandler/io/WReader.h"
#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMBemBoundary.h"

namespace LaBP
{

    class WReaderVOL: public WReader
    {
    public:

        struct ReturnCode
        {
            enum Enum
            {
                SUCCESS, /**< Normal */
                ERROR_FOPEN, /**< Error opening file */
                ERROR_FREAD, /**< File read error */
                ERROR_UNKNOWN /**< Unknown error */
            };
        };

        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WReaderVOL( std::string fname );

        ReturnCode::Enum read( boost::shared_ptr< std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > > > boundaries );

    private:
        ReturnCode::Enum readNumBoundaries( std::string& line, size_t& count );
        ReturnCode::Enum readConductUnit( std::string& line, WEUnit::Enum& unit );
        ReturnCode::Enum readConductivities( std::ifstream& ifs,
                        std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > >& boundaries );
        ReturnCode::Enum readBndFiles( std::ifstream& ifs, std::string& line,
                        std::vector< boost::shared_ptr< WDataSetEMMBemBoundary > >& boundaries );
    };
}
#endif /* WREADERVOL_H_ */
