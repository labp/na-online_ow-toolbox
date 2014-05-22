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

#include <modules/ftRtClient/fieldtrip/dataTypes/WFTEvent.h>

const std::string WFTEvent::CLASS = "WFTEvent";

WFTEvent::WFTEvent( WFTEventDefT def, const std::string type, const std::string value ) :
                m_def( def ), m_type( type ), m_value( value )
{

}

WFTEvent::WFTEvent( INT32_T sample, INT32_T offset, INT32_T duration, const std::string type, const std::string value ) :
                m_type( type ), m_value( value )
{
    m_def.sample = sample;
    m_def.offset = offset;
    m_def.duration = duration;
}

UINT32_T WFTEvent::getSize() const
{
    return ( UINT32_T )sizeof(eventdef_t) + m_def.bufsize;
}

WFTEventDefT& WFTEvent::getDef()
{
    return m_def;
}

WFTEventDefT WFTEvent::getDef() const
{
    return m_def;
}

const std::string WFTEvent::getType() const
{
    return m_type;
}

const std::string WFTEvent::getValue() const
{
    return m_value;
}
