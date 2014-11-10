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

#include <string>

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <message.h>

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftbClient/dataTypes/WFTEventList.h"

#include "../container/WFTEventIterator.h"
#include "../request/WFTRequest.h"
#include "../response/WFTResponse.h"

const std::string WFTEventList::CLASS = "WFTEventList";

WFTEventList::~WFTEventList()
{
}

WFTRequest::SPtr WFTEventList::asRequest()
{
    boost::shared_ptr< FtEventList > list( new FtEventList );

    BOOST_FOREACH( WFTEvent::SPtr event, *this )
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

    WFTEventIterator::SPtr it( new WFTEventIterator( &storage, response->getMessage().def->bufsize ) );

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

wftb::bufsize_t WFTEventList::getSize() const
{
    wftb::bufsize_t size = sizeof( wftb::MessageDefT );

    BOOST_FOREACH( WFTEvent::SPtr event, *this )
    {
        size += event->getSize();
    }

    return size;
}
