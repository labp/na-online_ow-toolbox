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

#ifndef WL2DCHANNELSCROLLHANDLER_H_
#define WL2DCHANNELSCROLLHANDLER_H_

#include <string>

#include <osg/ref_ptr>

#include <core/ui/WUIViewEventHandler.h>

#include "core/gui/drawable/WLEMDDrawable2DMultiChannel.h"

/**
 * Scrolls over the channels in the 2D view.
 *
 * \author pieloth
 * \ingroup gui
 */
class WL2DChannelScrollHandler: public WUIViewEventHandler
{
public:
    /**
     * Abbreviation for a osg::ref_ptr on a instance of this class.
     */
    typedef osg::ref_ptr< WL2DChannelScrollHandler > RefPtr;

    static const std::string CLASS;

    explicit WL2DChannelScrollHandler( WLEMDDrawable2DMultiChannel::SPtr initiator );

    virtual ~WL2DChannelScrollHandler();

    virtual void handleScroll( GUIEvents::ScrollingMotion motion, float deltaX, float deltaY );

    void setDrawable( WLEMDDrawable2DMultiChannel::SPtr drawable );

private:
    WLEMDDrawable2DMultiChannel::SPtr m_initiator;
};

#endif  // WL2DCHANNELSCROLLHANDLER_H_
