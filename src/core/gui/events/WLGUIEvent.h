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

#ifndef WLGUIEVENT_H_
#define WLGUIEVENT_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/any.hpp>

#include <osgGA/GUIEventAdapter>

/**
 * Base class for GUI events, which are used for notifications. Holds information about an event. GUI events can be fired by GUI elements like WLEMDDrawable.
 * Such class must provide addListener() methods.
 */
class WLGUIEvent
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLGUIEvent > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLGUIEvent > ConstSPtr;

    typedef std::string MiscTypeT;

    typedef boost::any ParamT;

    /**
     * Enum for event types. Events are grouped in classes aka type.
     */
    struct EventType
    {
        enum Enum
        {
            MOUSE, /*KEYBOARD, WINDOWS,*/MISC
        };
    };

    WLGUIEvent( EventType::Enum eventType, const osgGA::GUIEventAdapter& ea );
    virtual ~WLGUIEvent();

    EventType::Enum getEventType() const;

    void setMiscType( MiscTypeT name );

    MiscTypeT getMiscType() const;

    const osgGA::GUIEventAdapter& getOsgEventAdapter() const;

    /**
     * Gets an optional parameter, like a C-Union.
     */
    const ParamT& getParameter() const;

    /**
     * Sets an optional parameter, like a C-Union.
     */
    void setParameter( ParamT param );

    /**
     * Tries to cast a parameter to type T or throws an exception.
     */
    template< typename T >
    static T castParameter( const ParamT& param )
    {
        return boost::any_cast< T >( param );
    }

    /**
     * Tries to cast the parameter to type T or throws an exception.
     */
    template< typename T >
    T getParameterAs() const
    {
        return boost::any_cast< T >( m_param );
    }

private:
    const EventType::Enum m_eventType;

    const osgGA::GUIEventAdapter& m_eventAdapter;

    MiscTypeT m_miscType;

    ParamT m_param;
};

#endif  // WLGUIEVENT_H_
