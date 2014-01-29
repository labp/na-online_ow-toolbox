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

#include <string>

#include <core/common/WLogger.h>

#include "WL2DChannelScrollHandler.h"

const std::string WL2DChannelScrollHandler::CLASS = "WL2DChannelScrollHandler";

WL2DChannelScrollHandler::WL2DChannelScrollHandler( LaBP::WLEMDDrawable2DMultiChannel::SPtr initiator ) :
                WCustomWidgetEventHandler( initiator->getWidget() ), m_initiator( initiator )
{
    m_preselection |= GUIEvents::SCROLL;
}

WL2DChannelScrollHandler::~WL2DChannelScrollHandler()
{
}

void WL2DChannelScrollHandler::handleScroll( GUIEvents::ScrollingMotion motion, float deltaX, float deltaY )
{
    if( deltaY < 0.0 ) // down
    {
        const size_t channelNr = m_initiator->getChannelBegin();
        m_initiator->setChannelBegin( channelNr + 1 );
        return;
    }
    if( deltaY > 0.0 ) // up
    {
        const size_t channelNr = m_initiator->getChannelBegin();
        if( channelNr > 0 )
        {
            m_initiator->setChannelBegin( channelNr - 1 );
        }
        return;
    }
}

void WL2DChannelScrollHandler::setDrawable( LaBP::WLEMDDrawable2DMultiChannel::SPtr drawable )
{
    m_initiator = drawable;
}
