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

#ifndef WLROI_H_
#define WLROI_H_

#include <osg/Array>

#include <core/graphicsEngine/WPickInfo.h>
#include <core/graphicsEngine/WROI.h>
#include <core/ui/WUIViewWidget.h>

#include "core/gui/events/WLPickHandler.h"

/**
 * Wrapper class of WROI to do undone calculations.
 */
class WLROI: public WROI
{
public:

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WLROI.
     */
    WLROI( WUIViewWidget::SPtr widget );

    /**
     * Destroys the WLROI.
     */
    virtual ~WLROI();

    /**
     * Setter for the standard color.
     *
     * @param color The new color.
     */
    void setColor( osg::Vec4 color );

    /**
     * Setter for the color in negated state.
     *
     * @param color The new color.
     */
    void setNotColor( osg::Vec4 notColor );

protected:

    /**
     * Stores the pick information for potential redraw.
     */
    WPickInfo m_pickInfo;

    /**
     * The color of the box.
     */
    osg::Vec4 m_color;

    /**
     * The color of the box when negated.
     */
    osg::Vec4 m_notColor;

    /**
     * store this pointer for use in updates.
     */
    osg::ref_ptr< osg::Geometry > m_surfaceGeometry;

    /**
     * The widget.
     */
    WUIViewWidget::SPtr m_widget;

    /**
     * Lock to prevent concurrent threads trying to update the osg node.
     */
    boost::shared_mutex m_updateLock;

    /**
     * The pick handler for picking the ROI in a WLEMDDrawable view.
     */
    WLPickHandler::RefPtr m_mouseHandler;

    /**
     * Note that there was a pick.
     *
     * @param pickInfo Info from pick.
     */
    void registerRedrawRequest( WPickInfo pickInfo );

    /**
     * Set new color of the box in the geometry
     *
     * @param color the new color.
     */
    void updateColor( osg::Vec4 color );

private:

    /**
     *  Updates the graphics.
     */
    virtual void updateGFX() = 0;

};

#endif /* WLROI_H_ */
