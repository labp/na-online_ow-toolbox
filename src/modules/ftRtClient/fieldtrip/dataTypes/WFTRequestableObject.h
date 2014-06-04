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

#ifndef WFTREQUESTABLEOBJECT_H_
#define WFTREQUESTABLEOBJECT_H_

#include "modules/ftRtClient/fieldtrip/io/request/WFTRequest.h"
#include "modules/ftRtClient/fieldtrip/io/response/WFTResponse.h"
#include "WFTObject.h"

class WFTRequestableObject: public WFTObject
{
public:

    /**
     * Destroys the WFTRequestableObject.
     */
    virtual ~WFTRequestableObject();

    /**
     * Gets the object as FieldTrip put request.
     *
     * @return Returns a shared pointer on a WFTRequest.
     */
    virtual WFTRequest::SPtr asRequest() = 0;

    /**
     * Parses a WFTResponse into a concrete object.
     *
     * @param The response to parse.
     * @return Returns true if the parsing was successful, otherwise false.
     */
    virtual bool parseResponse( WFTResponse::SPtr ) = 0;

    /**
     * Gets the amount of bytes, which are reserved by the object. Each Fieldrip object has to determine its size itself.
     *
     * @return Returns an unsigned 32 bit integer.
     */
    virtual UINT32_T getSize() const = 0;
};

#endif /* WFTREQUESTABLEOBJECT_H_ */
