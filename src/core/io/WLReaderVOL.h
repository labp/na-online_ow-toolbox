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
#include "core/io/WLReaderGeneric.h"

/**
 * Reads BEM surfaces from FIF.
 *
 * \author pieloth
 * \ingroup io
 */
class WLReaderVOL: public WLReaderGeneric< std::list< WLEMMBemBoundary::SPtr > >
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

    virtual WLIOStatus::IOStatusT read( std::list< WLEMMBemBoundary::SPtr >* const boundaries );

private:
    WLIOStatus::IOStatusT readNumBoundaries( size_t* const count, const std::string& line );
    WLIOStatus::IOStatusT readConductUnit( WLEUnit::Enum* const unit, const std::string& line );
    WLIOStatus::IOStatusT readConductivities( std::ifstream& ifs, std::list< WLEMMBemBoundary::SPtr >* const boundaries );
    WLIOStatus::IOStatusT readBndFiles( std::ifstream& ifs, std::string* const line,
                    std::list< WLEMMBemBoundary::SPtr >* const boundaries );
};

#endif  // WLREADERVOL_H_
