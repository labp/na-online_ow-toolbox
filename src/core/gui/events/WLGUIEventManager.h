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

#ifndef WLGUIEVENTMANAGER_H_
#define WLGUIEVENTMANAGER_H_

#include <list>
#include <string>

#include <osgGA/GUIEventAdapter>

#include "WLGUIEventListener.h"
#include "WLGUIMouseEventListener.h"

class WLGUIEvent;
class WLGUIMouseEvent;

/**
 * A manager class for GUI events. This class should be used by GUI elements to provide a notification to listeners aka observer pattern.
 */
class WLGUIEventManager
{
public:
    static const std::string CLASS;

    WLGUIEventManager();
    virtual ~WLGUIEventManager();

    /**
     * Adds a listener to the general notification list.
     */
    void addEventListener( WLGUIEventListener::SPtr gel );

    /**
     * Removes a listener from the general notification list.
     */
    void removeEventListener( WLGUIEventListener::SPtr gel );

    /**
     * Clears the general notification list. This methods should be used to prevent memory leaks due to cyclic references.
     */
    void clearEventListeners();

    /**
     * Adds a mouse listener to the general notification list.
     */
    void addMouseEventListener( WLGUIMouseEventListener::SPtr gel );

    /**
     * Removes a mouse listener from the general notification list.
     */
    void removeMouseEventListener( WLGUIMouseEventListener::SPtr gel );

    /**
     * Clears the mouse notification list. This methods should be used to prevent memory leaks due to cyclic references.
     */
    void clearMouseEventListeners();

    /**
     * Clears all notification lists. This methods should be used to prevent memory leaks due to cyclic references.
     */
    void clearListeners();

    /**
     * Dispatchs a OSG event to the different GUI event types.
     */
    bool dispatchEvent( const osgGA::GUIEventAdapter& ea );

protected:
    /**
     * Sends a notification to all general listeners.
     */
    void notifyEventsListeners( const WLGUIEvent& e );

    /**
     * Sends a notification to all mouse listeners.
     */
    void notifyMouseEventsListeners( const WLGUIMouseEvent& e );

private:
    std::list< WLGUIEventListener::SPtr > m_eventListeners;
    std::list< WLGUIMouseEventListener::SPtr > m_mouseListeners;
};

#endif  // WLGUIEVENTMANAGER_H_
