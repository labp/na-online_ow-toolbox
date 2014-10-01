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

#include <core/common/WLogger.h>

#include "WL2DChannelScrollHandler.h"

const std::string WL2DChannelScrollHandler::CLASS = "WL2DChannelScrollHandler";

WL2DChannelScrollHandler::WL2DChannelScrollHandler( WLEMDDrawable2DMultiChannel::SPtr initiator ) :
                WUIViewEventHandler( initiator->getWidget() ), m_initiator( initiator )
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

void WL2DChannelScrollHandler::setDrawable( WLEMDDrawable2DMultiChannel::SPtr drawable )
{
    m_initiator = drawable;
}
