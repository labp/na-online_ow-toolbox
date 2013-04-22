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

#include <core/common/WLogger.h>
#include <core/gui/WCustomWidget.h>

#include "core/gui/drawable/WLEMDDrawable.h"

#include "WLGUIEventHandler.h"

namespace LaBP
{
    WLGUIEventHandler::WLGUIEventHandler( LaBP::WLEMDDrawable::SPtr initiator, LaBP::WLEMDDrawable::SPtr acceptor ) :
                    m_initiator( initiator ), m_acceptor( acceptor )
    {
        m_handlerDelegator = new WLGUIEventDelegator( this );
        m_initiator->getWidget()->getViewer()->getView()->addEventHandler( m_handlerDelegator );
    }

    WLGUIEventHandler::~WLGUIEventHandler()
    {
        // We need the delegator, because removeEventHandler calls the destructor too!
        m_initiator->getWidget()->getViewer()->getView()->removeEventHandler( m_handlerDelegator );
    }

    WLGUIEventHandler::WLGUIEventDelegator::WLGUIEventDelegator( osgGA::GUIEventHandler* handler ) :
                    m_handler( handler )
    {
    }

    WLGUIEventHandler::WLGUIEventDelegator::~WLGUIEventDelegator()
    {
    }

    bool WLGUIEventHandler::WLGUIEventDelegator::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
    {
        return m_handler->handle( ea, aa );
    }
}
