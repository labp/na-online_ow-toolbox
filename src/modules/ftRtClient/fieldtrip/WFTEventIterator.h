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

#ifndef WFTEVENTITERATOR_H_
#define WFTEVENTITERATOR_H_

#include <boost/shared_ptr.hpp>

#include "WFTAIterator.h"
#include "dataTypes/WFTEvent.h"

/**
 * The WFTChunkIterator can be used to run through a bulk of memory containing FieldTrip events.
 * This class has the standard iterator appearance with its characteristic operations.
 */
class WFTEventIterator: public WFTAIterator< WFTEvent >
{
public:

    /**
     * Represents the name of the class.
     */
    static const std::string CLASS;

    /**
     * A shared pointer on the iterator.
     */
    typedef boost::shared_ptr< WFTEventIterator > SPtr;

    /**
     * Constructs a new WFTEventIterator.
     *
     * @param buf A reference to the event storage memory.
     * @param size The size of the memory area.
     */
    WFTEventIterator( SimpleStorage& buf, int size );

    /**
     * Inherited method from WFTAIterator.
     *
     * @return Returns true if there are more events, else false.
     */
    bool hasNext() const;

    /**
     * Inherited method from WFTAIterator.
     *
     * @return Returns the next event.
     */
    WFTEvent::SPtr getNext();

};

#endif /* WFTEVENTITERATOR_H_ */