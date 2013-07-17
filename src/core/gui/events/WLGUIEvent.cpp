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

#include "WLGUIEvent.h"

WLGUIEvent::WLGUIEvent( EventType::Enum eventType, const osgGA::GUIEventAdapter& ea ) :
                m_eventType( eventType ), m_eventAdapter( ea )
{
}

WLGUIEvent::~WLGUIEvent()
{
}

WLGUIEvent::EventType::Enum WLGUIEvent::getEventType() const
{
    return m_eventType;
}

void WLGUIEvent::setMiscType( MiscTypeT name )
{
    m_miscType = name;
}

WLGUIEvent::MiscTypeT WLGUIEvent::getMiscType() const
{
    return m_miscType;
}

const osgGA::GUIEventAdapter& WLGUIEvent::getOsgEventAdapter() const
{
    return m_eventAdapter;
}

const WLGUIEvent::ParamT& WLGUIEvent::getParameter() const
{
    return m_param;
}

void WLGUIEvent::setParameter( ParamT param )
{
    m_param = param;
}
