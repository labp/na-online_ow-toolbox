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

#ifndef WLRESIZEHANDLER_H_
#define WLRESIZEHANDLER_H_

#include <string>

#include <osg/ref_ptr>

#include <core/ui/WUIViewEventHandler.h>

#include "core/gui/drawable/WLEMDDrawable.h"

/**
 * Catches a resize and force a redraw.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLResizeHandler: public WUIViewEventHandler
{
public:
    /**
     * Abbreviation for a osg::ref_ptr on a instance of this class.
     */
    typedef osg::ref_ptr< WLResizeHandler > RefPtr;

    static const std::string CLASS;

    explicit WLResizeHandler( WLEMDDrawable::SPtr drawable );
    virtual ~WLResizeHandler();

    virtual void handleResize( int xPos, int yPos, int width, int height );

    void setDrawable( WLEMDDrawable::SPtr drawable );

private:
    WLEMDDrawable::SPtr m_drawable;
};

#endif  // WLRESIZEHANDLER_H_
