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

#ifndef WLREADERDIGPOINTS_H_
#define WLREADERDIGPOINTS_H_

#include <string>
#include <list>

#include <boost/shared_ptr.hpp>

#include "core/data/WLDigPoint.h"
#include "core/io/WLReader.h"

class WLReaderDigPoints: public WLReader
{
public:
    static const std::string CLASS;

    /**
     * Convenience typedef for a boost::shared_ptr< WLDigPoint >
     */
    typedef boost::shared_ptr< WLReaderDigPoints > SPtr;

    /**
     * Convenience typedef for a  boost::shared_ptr< const WLDigPoint >
     */
    typedef boost::shared_ptr< const WLReaderDigPoints > ConstSPtr;

    explicit WLReaderDigPoints( std::string fname );
    virtual ~WLReaderDigPoints();

    ReturnCode::Enum read( std::list< WLDigPoint >* const out );
};

#endif  // WLREADERDIGPOINTS_H_
