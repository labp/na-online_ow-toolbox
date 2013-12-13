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

#ifndef WLREADERLEADFIELD_H_
#define WLREADERLEADFIELD_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/dataHandler/io/WReader.h>

#include "core/data/WLDataTypes.h"
#include "core/io/WLIOStatus.h"

class WLReaderLeadfield: public WReader, public WLIOStatus::WLIOStatusInterpreter
{
public:
    static const std::string CLASS;

    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLReaderLeadfield > SPtr;

    /**
     * Shared pointer abbreviation to a const instance of this class.
     */
    typedef boost::shared_ptr< const WLReaderLeadfield > ConstSPtr;

    explicit WLReaderLeadfield( std::string fname ) throw( WDHNoSuchFile );
    virtual ~WLReaderLeadfield();

    WLIOStatus::ioStatus_t read( WLMatrix::SPtr& leadfield );
};

#endif  // WLREADERLEADFIELD_H_