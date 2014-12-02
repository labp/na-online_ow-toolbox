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
 *
 * \author maschke
 * \ingroup gui
 */
class WLROIBox: public WLROI
{
public:
    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructs a new WLROIBox.
     *
     * \param minPos The min position.
     * \param maxPos The max position.
     */
    WLROIBox( WPosition minPos, WPosition maxPos, WUIViewWidget::SPtr widget );

    /**
     * Get the corner of the box that has minimal x, y and z values.
     *
     * \return Returns the corner position.
     */
    WPosition getMinPos() const;

    /**
     * Get the corner of the box that has maximal x, y and z values.
     *
     * \return Returns the corner position.
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
    static size_t maxBoxId; //!< Current maximum boxId over all boxes.

    size_t m_boxId; //!< Id of the current box.

    WPropGroup m_propGrp; //!< Group for box specific properties.

    WPropPosition m_minPos; //!< The minimum position of the box.

    WPosition m_minPosInit; //!< The initial minimal position.

    WPropPosition m_maxPos; //!< The maximum position of the box.

    WPosition m_maxPosInit; //!< The initial maximal position.

    WPropDouble m_width; //!< The box width property.

    WPropDouble m_height; //!< The box height property.

    WPropDouble m_depth; //!< The box depth property.

    WVector3d m_dimensions; //!< The box dimensions.

    WGEShader::RefPtr m_lightShader; //!< Shader for proper lighting.

    bool m_needVertexUpdate; //!< If true, the box' vertex data is updated.

    bool m_isPicked; //!< Indicates whether the box is currently picked or not.

    WPosition m_pickedPosition; //!< Caches the old picked position to a allow for comparison.

    WVector3d m_pickNormal; //!< Store the normal that occured when the pick action was started.

    WVector2d m_oldPixelPosition; //!< Caches the old picked position to a allow for comparison.

    int16_t m_oldScrollWheel; //!< Caches scroll wheel value.

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
         * \param node the osg node
         * \param nv the node visitor
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
     * \param property The property.
     */
    void boxPropertiesChanged( boost::shared_ptr< WPropertyBase > property );

    /**
     * Called when width, height or depth was changed.
     *
     * \param property The changed property.
     */
    void boxDimensionsChanged( boost::shared_ptr< WPropertyBase > property );
};

#endif  // WLROIBOX_H_
