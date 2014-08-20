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

#ifndef WLEMDDRAWABLE3DSOURCE_H_
#define WLEMDDRAWABLE3DSOURCE_H_

#include <list>

#include <boost/shared_ptr.hpp>
#include <boost/spirit/home/support/detail/hold_any.hpp>

#include <osg/Geode>
#include <osg/ShapeDrawable>

#include <core/common/WPropertyTypes.h>
#include <core/ui/WUIViewWidget.h>
#include <core/graphicsEngine/WPickInfo.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/WLEMMSurface.h"
#include "core/util/roi/WLROISelector.h"

#include "WLEMDDrawable3D.h"

class WLEMDDrawable3DSource: public WLEMDDrawable3D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3DSource > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3DSource > ConstSPtr;

    /**
     * A shared pointer on a WLROISelector< WLEMMSurface, list<size_t> >.
     */
    typedef boost::shared_ptr< WLROISelector< WLEMMSurface, std::list< size_t > > > ROISelectorSPtr;

    explicit WLEMDDrawable3DSource( WUIViewWidget::SPtr widget );

    virtual ~WLEMDDrawable3DSource();

    /**
     * Sets the ROI selector.
     *
     * @param roiSelector The ROI selector.
     */
    void setROISelector( ROISelectorSPtr roiSelector );

    /**
     * Gets the ROI selector.
     *
     * @return The ROI selector.
     */
    ROISelectorSPtr getROISelector();

protected:
    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    /**
     * The ROI selector.
     */
    ROISelectorSPtr m_roiSelecor;

private:
    void osgUpdateSurfaceColor( const WLEMData::DataT& data );

    /**
     * Callback function when a new ROI was added to the widget.
     */
    void callbackNewRoi_Clicked();

    void drawCoords();

    /**
     * The views properties.
     */
    boost::shared_ptr< WProperties > m_properties;

    /**
     * Trigger to create a new WLROI.
     */
    WPropTrigger m_trgNewRoi;

    osg::ref_ptr<osg::ShapeDrawable> m_shapeX;
    osg::ref_ptr<osg::ShapeDrawable> m_shapeY;
    osg::ref_ptr<osg::ShapeDrawable> m_shapeZ;
    osg::ref_ptr<osg::Geode> m_coords;
};

#endif  // WLEMDDRAWABLE3DSOURCE_H_
