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

#ifndef WLBADALLOCEXCEPTION_H_
#define WLBADALLOCEXCEPTION_H_

#include <string>

#include <core/common/WException.h>

/**
 * An exception that gets thrown on failure allocating memory, e.g. memory on a CUDA device.
 */
class WLBadAllocException: public WException
{
public:
    /**
     * Default constructor.
     * \param msg Exception description.
     */
    explicit WLBadAllocException( const std::string& msg = std::string() );

    /**
     * Destructor.
     */
    virtual ~WLBadAllocException() throw();
};

#endif  // WLBADALLOCEXCEPTION_H_
