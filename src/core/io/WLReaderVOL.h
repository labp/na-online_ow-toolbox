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

#ifndef WLREADERVOL_H_
#define WLREADERVOL_H_

#include <fstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMBemBoundary.h"
#include "core/io/WLReader.h"

namespace LaBP
{

    class WLReaderVOL: public WLReader
    {
    public:
        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WLReaderVOL( std::string fname );
        virtual ~WLReaderVOL()
        {
        }
        ;

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
#endif /* WLREADERVOL_H_ */
