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

#include "core/common/WLogger.h"

#include "dataTypes/WFTObject.h"
#include "dataTypes/WLEFTDataType.h"

#include "WFTEventIterator.h"

const std::string WFTEventIterator::CLASS = "WFTEventIterator";

WFTEventIterator::WFTEventIterator( SimpleStorage& buf, int size ) :
                WFTAIterator< WFTEvent >::WFTAIterator( buf, size )
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

    WFTObject::WFTEventDefT *def = ( WFTObject::WFTEventDefT * )( ( char * )m_store.data() + m_pos );
    unsigned int wsType, wsValue;

    // test whether the events is included completely
    if( m_pos + sizeof(WFTObject::WFTEventDefT) + def->bufsize < m_size )
    {
        return WFTEvent::SPtr();
    }

    // get the word sizes for type and value. Important for pointing to the right memory location.
    wsType = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->type_type ) );
    wsValue = WLEFTDataType::wordSize( WLEFTDataType::typeByCode( def->value_type ) );
    // define the lengths of type and value
    uint lenType = wsType * def->type_numel;
    uint lenValue = wsValue * def->value_numel;
    // create pointers to types and values location.
    const char *srcType = ( ( const char* )m_store.data() ) + m_pos + sizeof(WFTObject::WFTEventDefT);
    const char *srcValue = srcType + lenType;

    std::string type( srcType, lenType );
    std::string value( srcValue, lenValue );

//    wlog::debug( CLASS ) << "Sample: " << def->sample;
//    wlog::debug( CLASS ) << "Offset: " << def->offset;
//    wlog::debug( CLASS ) << "Duration: " << def->duration;
//    wlog::debug( CLASS ) << "Buffer Size: " << def->bufsize;
//    wlog::debug( CLASS ) << "Type, Elements: " << def->type_numel;
//    wlog::debug( CLASS ) << "Type, Type: " << def->type_type;
//    wlog::debug( CLASS ) << "Type, Length: " << lenType;
//    wlog::debug( CLASS ) << "Value, Elements: " << def->value_numel;
//    wlog::debug( CLASS ) << "Value, Type: " << def->value_type;
//    wlog::debug( CLASS ) << "Value, Length: " << lenValue;

    m_pos += sizeof(WFTObject::WFTEventDefT) + def->bufsize; // increase the position to the next event.

    WFTObject::WFTEventDefT d; // TODO(maschke): remove dummy variable
    d.sample = def->sample;
    return WFTEvent::SPtr( new WFTEvent( d, "sdsd", "dfdfd" ) ); // create the event object and return.
}
