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

class WFTEventIterator: public WFTAIterator< WFTEvent >
{
public:

    typedef boost::shared_ptr< WFTEventIterator > SPtr;

    WFTEventIterator( SimpleStorage& buf, int size );

    virtual bool hasNext() const;

    virtual void reset();

    virtual WFTEvent::SPtr getNext();
};

#endif /* WFTEVENTITERATOR_H_ */
