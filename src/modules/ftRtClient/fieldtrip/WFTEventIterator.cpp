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

#include "dataTypes/WFTObject.h"
#include "dataTypes/WLEFTDataType.h"

#include <modules/ftRtClient/fieldtrip/WFTEventIterator.h>

WFTEventIterator::WFTEventIterator( SimpleStorage& buf, int size ) :
                WFTAIterator( buf, size )
{

}

bool WFTEventIterator::hasNext() const
{
    return m_pos + sizeof(WFTObject::WFTEventDefT) < m_size;
}

void WFTEventIterator::reset()
{
    m_pos = 0;
}

WFTEvent::SPtr WFTEventIterator::getNext()
{
    if( !hasNext() )
    {
        return WFTEvent::SPtr();
    }

    const WFTObject::WFTEventDefT *def;
    unsigned int wsType, wsValue;

    // get the definition
    def = ( const WFTObject::WFTEventDefT * )( ( char * )m_store.data() + m_pos );

    // test whether the events is included completely
    if( m_pos + sizeof(WFTObject::WFTEventDefT) + def->bufsize < m_size )
    {
        return WFTEvent::SPtr();
    }

    wsType = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->type_type ) );
    wsValue = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->value_type ) );

    std::string *srcType = static_cast< std::string* >( ( std::string* )m_store.data() + m_pos + sizeof(WFTObject::WFTEventDefT) );
    std::string *srcValue = srcType + def->type_numel * wsType;

    std::string *type = (std::string *)malloc(wsType);
    std::string *value = (std::string *)malloc(wsValue);

    memcpy( type, srcType, wsType );
    memcpy( value, srcValue, wsValue );

    if( type == NULL || value == NULL )
    {
        return WFTEvent::SPtr();
    }

    m_pos += sizeof(WFTObject::WFTEventDefT) + def->bufsize; // increase the position to the next event.

    return WFTEvent::SPtr( new WFTEvent( *def, *type, *value ) );
}
