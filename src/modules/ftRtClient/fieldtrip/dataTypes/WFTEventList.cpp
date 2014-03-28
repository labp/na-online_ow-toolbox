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

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <message.h>

#include "core/common/WLogger.h"

#include <modules/ftRtClient/fieldtrip/WFTEventIterator.h>
#include <modules/ftRtClient/fieldtrip/dataTypes/WFTEventList.h>
#include <modules/ftRtClient/fieldtrip/io/request/WFTRequest.h>
#include <modules/ftRtClient/fieldtrip/io/response/WFTResponse.h>
#include <modules/ftRtClient/fieldtrip/dataTypes/WLEFTDataType.h>

const std::string WFTEventList::CLASS = "WFTEventList";

WFTRequest::SPtr WFTEventList::asRequest()
{
    boost::shared_ptr< FtEventList > list( new FtEventList );

    BOOST_FOREACH(WFTEvent::SPtr event, *this)
    {
        list->add( event->getDef().sample, event->getType().c_str(), event->getValue().c_str() );
    }

    WFTRequest::SPtr request( new WFTRequest( list->asRequest() ) );

    return request;
}

bool WFTEventList::parseResponse( WFTResponse::SPtr response )
{
    SimpleStorage storage;

    if( response->checkGetEvents( &storage ) < 0 )
    {
        return false;
    }

    WFTEventIterator::SPtr it( new WFTEventIterator( storage, response->getMessage().def->bufsize ) );

    while( it->hasNext() )
    {
        WFTEvent::SPtr evt = it->getNext();

        if( evt != NULL )
        {
            push_back( evt );
        }
    }

    return true;
}

UINT32_T WFTEventList::getSize() const
{
    UINT32_T size = sizeof(WFTMessageDefT);

    BOOST_FOREACH(WFTEvent::SPtr event, *this)
    {
        size += event->getSize();
    }

    return size;
}
