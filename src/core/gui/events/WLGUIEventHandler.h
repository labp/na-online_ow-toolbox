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

#ifndef WLGUIEVENTHANDLER_H_
#define WLGUIEVENTHANDLER_H_

#include <boost/shared_ptr.hpp>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

#include "core/gui/drawable/WLEMDDrawable.h"

namespace LaBP
{
    class WLGUIEventHandler: public osgGA::GUIEventHandler
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLGUIEventHandler > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLGUIEventHandler > ConstSPtr;

        WLGUIEventHandler( LaBP::WLEMDDrawable::SPtr initiator, LaBP::WLEMDDrawable::SPtr acceptor );
        virtual ~WLGUIEventHandler();
        virtual bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa ) = 0;

    protected:
        LaBP::WLEMDDrawable::SPtr m_initiator;
        LaBP::WLEMDDrawable::SPtr m_acceptor;

    private:
        /**
         * Event delegator class to prevent double destructor call of WLGUIEventHandler.
         */
        class WLGUIEventDelegator: public osgGA::GUIEventHandler
        {
        public:
            explicit WLGUIEventDelegator( osgGA::GUIEventHandler* handler );
            virtual ~WLGUIEventDelegator();

            virtual bool handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa );

        private:
            osgGA::GUIEventHandler* const m_handler;
        };

        /**
         * This handler is registered to the viewer instead of this-pointer. Otherwise the destructor is called twice!
         */
        WLGUIEventDelegator* m_handlerDelegator;
    };
}
#endif  // WLGUIEVENTHANDLER_H_
