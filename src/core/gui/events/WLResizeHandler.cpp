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

#include "WLResizeHandler.h"

const std::string WLResizeHandler::CLASS = "WLResizeHandler";

WLResizeHandler::WLResizeHandler( WLEMDDrawable::SPtr drawable ) :
                WUIViewEventHandler( drawable->getWidget() ), m_drawable( drawable )
{
    m_preselection |= GUIEvents::RESIZE;
}

WLResizeHandler::~WLResizeHandler()
{
}

void WLResizeHandler::handleResize( int xPos, int yPos, int width, int height )
{
    m_drawable->redraw();
}

void WLResizeHandler::setDrawable( WLEMDDrawable::SPtr drawable )
{
    m_drawable = drawable;
}
