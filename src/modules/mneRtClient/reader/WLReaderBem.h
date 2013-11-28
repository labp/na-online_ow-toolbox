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

#ifndef WLREADERBEM_H_
#define WLREADERBEM_H_

#include <string>
#include <list>

#include <boost/shared_ptr.hpp>

#include <core/dataHandler/io/WReader.h>

#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMEnumTypes.h"

class WLReaderBem: public WReader
{
public:
    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLReaderBem > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WLReaderBem > ConstSPtr;

    static const std::string CLASS;

    explicit WLReaderBem( std::string fname ) throw( WDHNoSuchFile );
    virtual ~WLReaderBem();

    bool read( std::list< LaBP::WLEMMBemBoundary::SPtr >* const bems );

private:
    static LaBP::WEBemType::Enum getTypeFromBemId(int id);
};

#endif  // WLREADERBEM_H_
