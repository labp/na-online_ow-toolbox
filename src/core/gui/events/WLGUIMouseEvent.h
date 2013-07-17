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

#ifndef WLGUIMOUSEEVENT_H_
#define WLGUIMOUSEEVENT_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "WLGUIEvent.h"

/**
 * Class for mouse event notifications.
 */
class WLGUIMouseEvent: public WLGUIEvent
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLGUIMouseEvent > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLGUIMouseEvent > ConstSPtr;

    struct Event
    {
        enum Enum
        {
            CLICK_LEFT, CLICK_MIDDLE, CLICK_CENTER, SCROLL_UP, SCROLL_DOWN, MISC
        };
    };

    WLGUIMouseEvent( Event::Enum event, const osgGA::GUIEventAdapter& ea );
    virtual ~WLGUIMouseEvent();

    Event::Enum getEvent() const;

    void setMiscEvent( std::string miscEvent );

    std::string getMiscEvent() const;

private:
    const Event::Enum m_event;

    std::string m_miscEvent;
};

#endif  // WLGUIMOUSEEVENT_H_
