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

#include "modules/ftRtClient/ftb/WFtbEvent.h"

#include "WFTEventIterator.h"

const std::string WFTEventIterator::CLASS = "WFTEventIterator";

WFTEventIterator::WFTEventIterator( SimpleStorage* const buf, int size ) :
                WFTAIterator< WFTEvent >::WFTAIterator( buf, size )
{
}

bool WFTEventIterator::hasNext() const
{
    return m_pos + ( int )sizeof( wftb::EventDefT ) < m_size;
}

WFTEvent::SPtr WFTEventIterator::getNext()
{
    if( !hasNext() )
    {
        return WFTEvent::SPtr();
    }

    wftb::EventDefT *def = ( wftb::EventDefT * )( ( char * )m_store->data() + m_pos );
    unsigned int wsType, wsValue;

    // test whether the events is included completely
    if( m_pos + ( int )( sizeof( wftb::EventDefT ) + def->bufsize ) > m_size )
    {
        return WFTEvent::SPtr();
    }

    // get the word sizes for type and value. Important for pointing to the right memory location.
    wsType = sizeof( wftb::Event::type_type_t );
    wsValue = sizeof( wftb::Event::value_type_t );
    // define the lengths of type and value
    uint lenType = wsType * def->type_numel;
    uint lenValue = wsValue * def->value_numel;
    // create pointers to types and values location.
    const char *srcType = ( ( const char* )m_store->data() ) + m_pos + sizeof( wftb::EventDefT );
    const char *srcValue = srcType + lenType;

    std::string type( srcType, lenType );
    std::string value( srcValue, lenValue );

    m_pos += sizeof( wftb::EventDefT ) + def->bufsize; // increase the position to the next event.

    return WFTEvent::SPtr( new WFTEvent( *def, type, value ) ); // create the event object and return.
}
