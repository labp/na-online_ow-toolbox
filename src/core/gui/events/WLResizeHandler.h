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

#ifndef WLRESIZEHANDLER_H_
#define WLRESIZEHANDLER_H_

#include <string>

#include <osg/ref_ptr>

#include <core/ui/WCustomWidgetEventHandler.h>

#include "core/gui/drawable/WLEMDDrawable.h"

/**
 * Catches a resize and force a redraw.
 *
 * \author pieloth
 */
class WLResizeHandler: public WCustomWidgetEventHandler
{
public:
    /**
     * Abbreviation for a osg::ref_ptr on a instance of this class.
     */
    typedef osg::ref_ptr< WLResizeHandler > RefPtr;

    static const std::string CLASS;

    WLResizeHandler( LaBP::WLEMDDrawable::SPtr drawable );
    virtual ~WLResizeHandler();

    virtual void handleResize( int xPos, int yPos, int width, int height );

    void setDrawable( LaBP::WLEMDDrawable::SPtr drawable );

private:
    LaBP::WLEMDDrawable::SPtr m_drawable;
};

#endif  // WLRESIZEHANDLER_H_
