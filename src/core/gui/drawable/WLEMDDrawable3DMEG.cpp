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

#include <core/gui/WCustomWidget.h>

#include "WLEMDDrawable3D.h"
#include "WLEMDDrawable3DMEG.h"

namespace LaBP
{
    WLEMDDrawable3DMEG::WLEMDDrawable3DMEG( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable3D( widget )
    {
    }

    WLEMDDrawable3DMEG::~WLEMDDrawable3DMEG()
    {
    }

    void WLEMDDrawable3DMEG::redraw()
    {
        if( !hasData() )
        {
            return;
        }
        clearWidget();
        // TODO(pizarro): code here
        updateWidget();
    }

    void WLEMDDrawable3DMEG::osgNodeCallback( osg::NodeVisitor* nv )
    {
        // TODO(pizarro): modify scene graph here
    }
} /* namespace LaBP */
