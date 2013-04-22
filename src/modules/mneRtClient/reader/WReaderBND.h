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

#ifndef WREADERBND_H_
#define WREADERBND_H_

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/dataHandler/WDataSetEMMBemBoundary.h"

#include "core/dataHandler/io/WReader.h"

namespace LaBP
{

    class WReaderBND: public WReader
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
        explicit WReaderBND( std::string fname );

        ReturnCode::Enum read( boost::shared_ptr< WDataSetEMMBemBoundary > boundary );

    private:
        ReturnCode::Enum readType( std::string& line, boost::shared_ptr< WDataSetEMMBemBoundary > boundary );
        ReturnCode::Enum readUnit( std::string& line, boost::shared_ptr< WDataSetEMMBemBoundary > boundary );
        ReturnCode::Enum readNumPos( std::string& line, size_t& count );
        ReturnCode::Enum readNumPoly( std::string& line, size_t& count );
        ReturnCode::Enum readPositions( std::ifstream& ifs, size_t count, boost::shared_ptr< WDataSetEMMBemBoundary > boundary );
        ReturnCode::Enum readPolygons( std::ifstream& ifs, size_t count, boost::shared_ptr< WDataSetEMMBemBoundary > boundary );
    };
}
#endif /* WREADERBND_H_ */
