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

#ifndef WREADERVOL_H
#define WREADERVOL_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/dataHandler/io/WReader.h>
#include <core/dataHandler/WDataSet.h>


/**
 * Read data from a VOL file.
 * \ingroup dataHandler
 */
class WReaderVOL : public WReader // NOLINT
{
public:
    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WReaderVOL( std::string fname );

    /**
     * Read the file and create a dataset out of it.
     *
     * \return reference to the dataset
     */
    std::vector< std::string > read();


protected:
    std::vector< std::string > filenameBoundaries;
private:
};


#endif  // WREADERVOL_H
