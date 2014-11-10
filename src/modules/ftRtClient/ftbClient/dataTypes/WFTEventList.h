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

#ifndef WFTEVENTLIST_H_
#define WFTEVENTLIST_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "core/container/WLArrayList.h"

#include "WFTEvent.h"
#include "WFTRequestableObject.h"

class WFTEventList: public WLArrayList< WFTEvent::SPtr >, public WFTRequestableObject
{
public:
    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * A shared pointer on an event list.
     */
    typedef boost::shared_ptr< WFTEventList > SPtr;

    /**
     * Destroys th WFTEventList.
     */
    virtual ~WFTEventList();

    /**
     * Inherited from WFTRequestableObject.
     *
     * \return The event list as FieldTrip request.
     */
    WFTRequest::SPtr asRequest();

    /**
     * Inherited from WFTRequestableObject.
     *
     * \param The response to parse.
     * \return True if the parsing was successful, else false.
     */
    bool parseResponse( WFTResponse::SPtr );

    /**
     * Inherited from WFTRequestableObject
     *
     * \return The whole size of the object including definition and buffer.
     */
    UINT32_T getSize() const;
};

#endif  // WFTEVENTLIST_H_
