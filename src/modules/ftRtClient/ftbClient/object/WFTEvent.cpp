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

#include "WFTEvent.h"

const std::string WFTEvent::CLASS = "WFTEvent";

WFTEvent::WFTEvent( wftb::EventDefT def, const std::string type, const std::string value ) :
                m_def( def ), m_type( type ), m_value( value )
{
    m_def.sample = 0;
    m_def.offset = 0;
    m_def.duration = 0;
}

WFTEvent::~WFTEvent()
{
}

wftb::bufsize_t WFTEvent::getSize() const
{
    return ( wftb::bufsize_t )sizeof(eventdef_t) + m_def.bufsize;
}

wftb::bufsize_t WFTEvent::getDataSize() const
{
    return m_def.bufsize;
}

const wftb::EventDefT& WFTEvent::getDef() const
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
