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

#ifndef WREADEREEGPOSITIONS_H_
#define WREADEREEGPOSITIONS_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/dataHandler/exceptions/WDHNoSuchFile.h>

#include "core/io/WLReaderGeneric.h"

class WReaderEEGPositions: public WLReaderGeneric< std::vector< WPosition > >
{
public:
    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WReaderEEGPositions > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WReaderEEGPositions > ConstSPtr;

    static const std::string CLASS;

    explicit WReaderEEGPositions( std::string fname ) throw( WDHNoSuchFile );

    virtual ~WReaderEEGPositions();

    virtual WLIOStatus::IOStatusT read( std::vector< WPosition >* const positions );
};

#endif  // WREADEREEGPOSITIONS_H_
