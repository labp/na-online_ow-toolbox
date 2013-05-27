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

#include <list>
#include <string>

#include <core/common/WLogger.h>

#include "WLGUIEvent.h"
#include "WLGUIMouseEvent.h"
#include "WLGUIEventListener.h"
#include "WLGUIMouseEventListener.h"

#include "WLGUIEventManager.h"

const std::string WLGUIEventManager::CLASS = "WLGUIEventManager";

WLGUIEventManager::WLGUIEventManager()
{
}

WLGUIEventManager::~WLGUIEventManager()
{
}

void WLGUIEventManager::addEventListener( WLGUIEventListener::SPtr gel )
{
    m_eventListeners.push_back( gel );
}

void WLGUIEventManager::removeEventListener( WLGUIEventListener::SPtr gel )
{
    m_eventListeners.remove( gel );
}

void WLGUIEventManager::clearEventListeners()
{
    m_eventListeners.clear();
}

void WLGUIEventManager::addMouseEventListener( WLGUIMouseEventListener::SPtr gel )
{
    m_mouseListeners.push_back( gel );
}

void WLGUIEventManager::removeMouseEventListener( WLGUIMouseEventListener::SPtr gel )
{
    m_mouseListeners.remove( gel );
}

void WLGUIEventManager::clearMouseEventListeners()
{
    m_mouseListeners.clear();
}

void WLGUIEventManager::clearListeners()
{
    m_eventListeners.clear();
    m_mouseListeners.clear();
}

void WLGUIEventManager::notifyEventsListeners( const WLGUIEvent& e )
{
    wlog::debug( CLASS ) << "notifyEventsListeners() called";
    std::list< WLGUIEventListener::SPtr >::iterator it = m_eventListeners.begin();
    for( ; it != m_eventListeners.end(); ++it )
    {
        ( *it )->eventOccurred( e );
    }
}
void WLGUIEventManager::notifyMouseEventsListeners( const WLGUIMouseEvent& e )
{
    wlog::debug( CLASS ) << "notifyMouseEventsListeners() called";
    std::list< WLGUIMouseEventListener::SPtr >::iterator it = m_mouseListeners.begin();
    for( ; it != m_mouseListeners.end(); ++it )
    {
        ( *it )->mouseEventOccurred( e );
    }
}

bool WLGUIEventManager::dispatchEvent( const osgGA::GUIEventAdapter& ea )
{
    const osgGA::GUIEventAdapter::EventType osgEventType = ea.getEventType();
    const int osgButton = ea.getButton();

    if( !m_eventListeners.empty() )
    {
        // TODO(pieloth)
        WLGUIEvent e( WLGUIEvent::EventType::MISC, ea );
        notifyEventsListeners( e );
    }
    if( !m_mouseListeners.empty() )
    {
        if( osgEventType == osgGA::GUIEventAdapter::PUSH && osgButton == osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON )
        {
            WLGUIMouseEvent e( WLGUIMouseEvent::Event::CLICK_LEFT, ea );
            notifyMouseEventsListeners( e );
            return true;
        }
        if( osgEventType == osgGA::GUIEventAdapter::SCROLL )
        {
            // Workaround: ea.getScrollingMotion() == osgGA::GUIEventAdapter::SCROLL_DOWN / SCROLL_UP
            const float y_delta = ea.getScrollingDeltaY();
            if( y_delta != 0 )
            {
                WLGUIMouseEvent::Event::Enum event;
                if( y_delta < 0 ) // down
                {
                    event = WLGUIMouseEvent::Event::SCROLL_DOWN;
                }
                if( y_delta > 0 ) // up
                {
                    event = WLGUIMouseEvent::Event::SCROLL_UP;
                }
                WLGUIMouseEvent e( event, ea );
                notifyMouseEventsListeners( e );
                return true;
            }
        }
    }
    return false;
}
