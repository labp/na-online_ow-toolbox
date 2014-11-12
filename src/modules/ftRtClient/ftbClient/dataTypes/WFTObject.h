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

#ifndef WFTOBJECT_H_
#define WFTOBJECT_H_

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/ftb/WFtBuffer.h"

/**
 * The WFTObject interface provides basic types covered from the FieldTrip library and general methods for a processing object.
 */
class WFTObject
{
public:
    /**
     * A shared pointer on a WFTObject.
     */
    typedef boost::shared_ptr< WFTObject > SPtr;

    /**
     * Destroys the WFTObject.
     */
    virtual ~WFTObject();

    /**
     * Gets the objects memory size including all header data.
     *
     * \return The memory size.
     */
    virtual wftb::bufsize_t getSize() const = 0;
};

#endif  // WFTOBJECT_H_
