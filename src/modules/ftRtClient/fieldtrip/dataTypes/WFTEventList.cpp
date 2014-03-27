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

//    WFTEventIterator::SPtr it( new WFTEventIterator( *(buf.get()), response->getMessage()->def->bufsize ) );

    unsigned int pos = 0;

    while( pos + sizeof(WFTObject::WFTEventDefT) < storage.size() )
    {
        WFTObject::WFTEventDefT *def = ( WFTObject::WFTEventDefT * )( ( ( char * )storage.data() ) + pos );
        unsigned int wsType, wsValue;

        wsType = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->type_type ) );
        wsValue = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->value_type ) );

        uint lenType = wsType * def->type_numel;
        uint lenValue = wsValue * def->value_numel;
//            create pointers to types and values location.
        const char *srcType = ( ( const char* )storage.data() ) + pos + sizeof(WFTObject::WFTEventDefT);
        const char *srcValue = srcType + lenType;

        std::string type( srcType, lenType );
        std::string value( srcValue, lenValue );

        pos += sizeof(WFTObject::WFTEventDefT) + def->bufsize;

        push_back( WFTEvent::SPtr( new WFTEvent( *def, type, value ) ) );
    }

    // TODO(maschke): Why does not work the WFTEventIterator?
//    while( it->hasNext() )
//    {
//        push_back( it->getNext() );
//    }

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
