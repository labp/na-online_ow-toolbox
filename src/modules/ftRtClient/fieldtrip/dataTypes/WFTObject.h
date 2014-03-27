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

#include <boost/shared_ptr.hpp>

#include <message.h>

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
     * A FieldTrip Buffer message.
     */
    typedef message_t WFTMessageT;

    /**
     * A shared pointer on a constant FieldTrip Buffer message.
     */
    typedef boost::shared_ptr< const WFTMessageT > WFTMessageT_ConstSPtr;

    /**
     * A message header.
     */
    typedef messagedef_t WFTMessageDefT;

    /**
     * A FieldTrip header structure.
     */
    typedef header_t WFTHeaderT;

    /**
     * The fixed describing part of a FieldTrip header structure.
     */
    typedef headerdef_t WFTHeaderDefT;

    /**
     * A FieldTrip data object.
     */
    typedef data_t WFTDataT;

    /**
     * A header of a FieldTrip data object.
     */
    typedef datadef_t WFTDataDefT;

    /**
     * A FieldTrip event.
     */
    typedef event_t WFTEventT;

    /**
     * A header of a FieldTrip event.
     */
    typedef eventdef_t WFTEventDefT;

    /**
     * The structure for doing Wait-Requests.
     */
    typedef waitdef_t WFTWaitDefT;

    /**
     * A FieldTrip header chunk.
     */
    typedef ft_chunk_t WFTChunkT;

    /**
     * A shared pointer on a FieldTrip header chunk.
     */
    typedef boost::shared_ptr< WFTChunkT > WFTChunkT_SPtr;

    /**
     * A header of a FieldTrip chunk.
     */
    typedef ft_chunkdef_t WFTChunkDefT;

    /**
     * A shared pointer on a FieldTrip chunk header.
     */
    typedef boost::shared_ptr< ft_chunkdef_t > WFTChunkDefT_SPtr;

    /**
     * A structure to indicating a range of samples used by a data request.
     */
    typedef datasel_t WFTDataSelectionT;

    /**
     * A structure, which indicates during a Wait-request, how many samples and events already read by the client.
     */
    typedef samples_events_t WFTSamplesEventsT;

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
