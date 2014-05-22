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

#ifndef WFTOBJECT_H_
#define WFTOBJECT_H_

#include "WFTDataTypes.h"

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
     * @return The memory size.
     */
    virtual UINT32_T getSize() const = 0;
};

#endif /* WFTOBJECT_H_ */
