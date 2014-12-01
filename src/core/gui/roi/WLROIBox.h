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

#ifndef WLROIBOX_H_
#define WLROIBOX_H_

#include <boost/thread.hpp>

#include <osg/Array>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/graphicsEngine/shaders/WGEShader.h>
#include <core/graphicsEngine/WPickHandler.h>
#include <core/graphicsEngine/WROIBox.h>

#include "WLROI.h"

/**
 * The representation of a Region of Interest as a box volume.
 */
class WLROIBox: public WLROI
{
public:

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WLROIBox.
     *
     * @param minPos The min position.
     * @param maxPos The max position.
     */
    WLROIBox( WPosition minPos, WPosition maxPos, WUIViewWidget::SPtr widget );

    /**
     * Get the corner of the box that has minimal x, y and z values.
     *
     * @return Returns the corner position.
     */
    WPosition getMinPos() const;

    /**
     * Get the corner of the box that has maximal x, y and z values.
     *
     * @return Returns the corner position.
     */
    WPosition getMaxPos() const;

protected:

    /**
     * Destroys the WLROIBox.
     */
    virtual ~WLROIBox();

    /**
     * Initialize the box properties.
     */
    virtual void initProperties();

private:

    /**
     * Current maximum boxId over all boxes.
     */
    static size_t maxBoxId;

    /**
     * Id of the current box.
     */
    size_t m_boxId;

    /**
     * Group for box specific properties.
     */
    WPropGroup m_propGrp;

    /**
     * The minimum position of the box.
     */
    WPropPosition m_minPos;

    /**
     * The initial minimal position.
     */
    WPosition m_minPosInit;

    /**
     * The maximum position of the box.
     */
    WPropPosition m_maxPos;

    /**
     * The initial maximal position.
     */
    WPosition m_maxPosInit;

    /**
     * The box width property.
     */
    WPropDouble m_width;

    /**
     * The box height property.
     */
    WPropDouble m_height;

    /**
     * The box depth property.
     */
    WPropDouble m_depth;

    /**
     * The box dimensions.
     */
    WVector3d m_dimensions;

    /**
     * Shader for proper lighting.
     */
    WGEShader::RefPtr m_lightShader;

    /**
     * If true, the box' vertex data is updated.
     */
    bool m_needVertexUpdate;

    /**
     * Indicates whether the box is currently picked or not.
     */
    bool m_isPicked;

    /**
     * Caches the old picked position to a allow for comparison.
     */
    WPosition m_pickedPosition;

    /**
     * Store the normal that occured when the pick action was started.
     */
    WVector3d m_pickNormal;

    /**
     * Caches the old picked position to a allow for comparison.
     */
    WVector2d m_oldPixelPosition;

    /**
     * caches scroll wheel value.
     */
    int16_t m_oldScrollWheel;

    /**
     *  Updates the graphics.
     */
    virtual void updateGFX();

    /**
     * Node callback to handle updates properly.
     */
    class ROIBoxNodeCallback: public osg::NodeCallback
    {
    public:
        // NOLINT
        /**
         * operator ()
         *
         * @param node the osg node
         * @param nv the node visitor
         */
        virtual void operator()( osg::Node* node, osg::NodeVisitor* nv )
        {
            osg::ref_ptr< WLROIBox > module = static_cast< WLROIBox* >( node->getUserData() );
            if( module )
            {
                module->updateGFX();
            }
            traverse( node, nv );
        }
    };

    /**
     * Called when the specified property has changed. Used to update the ROI when modifying box properties.
     *
     * @param property The property.
     */
    void boxPropertiesChanged( boost::shared_ptr< WPropertyBase > property );

    /**
     * Called when width, height or depth was changed.
     *
     * @param property The changed property.
     */
    void boxDimensionsChanged( boost::shared_ptr< WPropertyBase > property );
};

#endif /* WLROIBOX_H_ */
