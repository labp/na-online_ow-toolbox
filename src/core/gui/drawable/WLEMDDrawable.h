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

#ifndef WLEMDDRAWABLE_H_
#define WLEMDDRAWABLE_H_

#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <osg/Node>
#include <osg/NodeCallback>
#include <osg/NodeVisitor>
#include <osg/ref_ptr>

#include <core/graphicsEngine/WGEGroupNode.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"
#include "core/util/WLDefines.h"

/**
 * An abstract class to draw EMD data into a widget.
 * It uses an internal callback function to modify the OSG tree in a single-threaded manner.
 *
 * \attention All functions which access and modify the OSG tree must have the prefix osg
 * and must be called from osgNodeCallback()! Otherwise OSG could produce a segmentation fault.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable: public boost::enable_shared_from_this< WLEMDDrawable >
{
public:
    typedef boost::shared_ptr< WLEMDDrawable > SPtr; //!< Abbreviation for a shared pointer on a instance of this class.

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable > ConstSPtr; //!<

    typedef float ValueT; //!< Type for all scalars x, y, z. Similar to osg::value_type.

    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructor
     *
     * \param widget Widget to fill.
     */
    explicit WLEMDDrawable( WUIViewWidget::SPtr widget );

    /**
     * Destructor
     */
    virtual ~WLEMDDrawable();

    /**
     * Cast to DRAWABLE if possible.
     *
     * \return Shared Pointer< DRAWABLE >
     */
    template< typename DRAWABLE >
    boost::shared_ptr< DRAWABLE > getAs()
    {
        return boost::dynamic_pointer_cast< DRAWABLE >( shared_from_this() );
    }

    /**
     * Cast to DRAWABLE if possible.
     *
     * \return Shared Pointer< const DRAWABLE >
     */
    template< typename DRAWABLE >
    boost::shared_ptr< const DRAWABLE > getAs() const
    {
        return boost::dynamic_pointer_cast< DRAWABLE >( shared_from_this() );
    }

    /**
     * Invokes a draw with the new data.
     *
     * \param emm Data to draw.
     */
    virtual void draw( WLEMMeasurement::SPtr emm ) = 0;

    /**
     * Invokes a draw with the last data.
     */
    virtual void redraw();

    /**
     * Checks if data is available.
     *
     * \return True if data is available.
     */
    virtual bool hasData() const = 0;

    /**
     * Gets modality to draw.
     *
     * \return modality
     */
    virtual WLEModality::Enum getModality() const;

    /**
     * Sets modality to draw.
     *
     * \return True if modality was changed.
     */
    virtual bool setModality( WLEModality::Enum modality );

    /**
     * Returns the widget to fill.
     *
     * \return widget Widget to fill.
     */
    virtual WUIViewWidget::SPtr getWidget() const;

    virtual float getSelectedTime() const = 0;

    virtual bool setSelectedTime( float relative ) = 0;

protected:
    /**
     * Implements the visualization of the data. Modifies and draws the scene graph.
     * This method is called as a callback function by the OSG system.
     * \note Modifications on the widget are allowed in this method (and sub calls) only, ensure single threaded access!
     *
     * \param nv osg::NodeVisitor
     */
    virtual void osgNodeCallback( osg::NodeVisitor* nv ) = 0;

    /**
     * \brief Checks if a new draw has to be invoked.
     * Checks if a new draw has to be invoked, e.g. data or modality  were changed.
     * \note This method should be overwritten and called from childs.
     *
     * \return True if a new draw is necessary.
     */
    virtual bool mustDraw() const;

    /**
     * Resets all draw flags, see  mustDraw().
     * \note This method should be overwritten and called from childs.
     */
    virtual void resetDrawFlags();

    /**
     * Root node for all OSG objects which are added to the widget. osgNodeCallback() is applied to this node.
     */
    osg::ref_ptr< WGEGroupNode > m_rootGroup;

    WLEModality::Enum m_modality; //!< Modality to draw.

    bool m_dataChanged; //!< A flag to indicate a data change.

    bool m_modalityChanged; //!< A flag to indicate a change of modality.

    /**
     * Widget to fill.
     * \attention If we change the widget, than we have a dangling event handler on the old widget.
     * This can cause some problems - remove and safely release all event handler.
     */
    const WUIViewWidget::SPtr m_widget;

private:
    bool m_draw; //!< A flag to indicate osgNodeCallback() that the data should be draw, used by redraw().

    /**
     * A wrapper of WLEMDDrawable to register it as a callback for m_rootGroup.
     * This approach is used to avoid unnamed callback functions from osg::NodeCallback.
     * Instead all drawables must implement draw() and osgNodeCallback().
     */
    class WLEMDDrawableCallbackDelegator: public osg::NodeCallback
    {
    public:
        /**
         * Abbreviation for a osg::ref_ptr on a instance of this class.
         */
        typedef osg::ref_ptr< WLEMDDrawableCallbackDelegator > RefPtr;

        /**
         * Constructor
         *
         * \param drawable Object to delegate the callback to.
         */
        explicit WLEMDDrawableCallbackDelegator( WLEMDDrawable* drawable );

        /**
         * Destructor.
         */
        virtual ~WLEMDDrawableCallbackDelegator();

        /**
         * Calls WLEMDDrawable::osgNodeCallback() of the wrapped drawable.
         *
         * \param node osg::Node*
         * \param nv osg::NodeVisitor*
         */
        void operator()( osg::Node* node, osg::NodeVisitor* nv );

        WLEMDDrawable* m_drawable; //!< WLEMDDrawable to wrap.
    };

    /**
     * Wrapper to register an instance of this class as a callback to the OSG root node.
     * The delegator is added in the constructor to the OSG root node via osg::Node::addUpdateCallback()
     * and removed in the destructor via osg::Node::removeUpdateCallback().
     */
    WLEMDDrawableCallbackDelegator::RefPtr m_callbackDelegator;
};

inline void WLEMDDrawable::WLEMDDrawableCallbackDelegator::operator()( osg::Node* node, osg::NodeVisitor* nv )
{
    WL_UNUSED( node );
    if( m_drawable != NULL )
    {
        m_drawable->osgNodeCallback( nv );
    }
}

#endif  // WLEMDDRAWABLE_H_
