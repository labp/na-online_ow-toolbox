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

#include "core/util/WLDefines.h"

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
    WL_UNUSED( xPos );
    WL_UNUSED( yPos );
    WL_UNUSED( width );
    WL_UNUSED( height );
    m_drawable->redraw();
}

void WLResizeHandler::setDrawable( WLEMDDrawable::SPtr drawable )
{
    m_drawable = drawable;
}
