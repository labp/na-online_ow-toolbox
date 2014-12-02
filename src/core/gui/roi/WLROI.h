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

#ifndef WLROI_H_
#define WLROI_H_

#include <osg/Array>

#include <core/graphicsEngine/WPickInfo.h>
#include <core/graphicsEngine/WROI.h>
#include <core/ui/WUIViewWidget.h>

#include "core/gui/events/WLPickHandler.h"

/**
 * Wrapper class of WROI to do undone calculations.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup gui
 */
class WLROI: public WROI
{
public:
    static const std::string CLASS; //!< Class name for logging purpose.

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
     * \param color The new color.
     */
    void setColor( osg::Vec4 color );

    /**
     * Setter for the color in negated state.
     *
     * \param color The new color.
     */
    void setNotColor( osg::Vec4 notColor );

protected:
    WPickInfo m_pickInfo; //!< Stores the pick information for potential redraw.

    osg::Vec4 m_color; //!< The color of the box.

    osg::Vec4 m_notColor; //!< The color of the box when negated.

    osg::ref_ptr< osg::Geometry > m_surfaceGeometry; //!< Store this pointer for use in updates.

    WUIViewWidget::SPtr m_widget; //!< The widget.

    boost::shared_mutex m_updateLock; //!< Lock to prevent concurrent threads trying to update the osg node.

    WLPickHandler::RefPtr m_mouseHandler; //!< The pick handler for picking the ROI in a WLEMDDrawable view.

    /**
     * Note that there was a pick.
     *
     * \param pickInfo Info from pick.
     */
    void registerRedrawRequest( WPickInfo pickInfo );

    /**
     * Set new color of the box in the geometry
     *
     * \param color the new color.
     */
    void updateColor( osg::Vec4 color );

private:
    /**
     *  Updates the graphics.
     */
    virtual void updateGFX() = 0;
};

#endif  // WLROI_H_
