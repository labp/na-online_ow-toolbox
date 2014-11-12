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

#ifndef WFTREQUESTABLEOBJECT_H_
#define WFTREQUESTABLEOBJECT_H_

#include "../WFTRequest.h"
#include "../WFTResponse.h"
#include "WFTObject.h"

class WFTRequestableObject: public WFTObject
{
public:
    /**
     * Destroys the WFTRequestableObject.
     */
    virtual ~WFTRequestableObject();

    /**
     * Parses a WFTResponse into a concrete object.
     *
     * \param The response to parse.
     * \return Returns true if the parsing was successful, otherwise false.
     */
    virtual bool parseResponse( const WFTResponse& ) = 0;

    /**
     * Gets the amount of bytes, which are reserved by the object. Each Fieldrip object has to determine its size itself.
     *
     * \return Returns an unsigned 32 bit integer.
     */
    virtual wftb::bufsize_t getSize() const = 0;
};

#endif  // WFTREQUESTABLEOBJECT_H_
