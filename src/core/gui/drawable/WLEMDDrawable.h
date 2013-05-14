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

#ifndef WLEMDDRAWABLE_H_
#define WLEMDDRAWABLE_H_

#include <cstddef>
#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <osg/Group>
#include <osg/Node>
#include <osg/NodeCallback>
#include <osg/NodeVisitor>
#include <osg/ref_ptr>

#include <core/gui/WCustomWidget.h>

#include "core/data/WLDataSetEMM.h"
#include "core/data/WLEMMEnumTypes.h"

namespace LaBP
{
    /**
     * An abstract class to draw EMD data into a widget.
     * It uses an internal callback function to modify the OSG tree in a single-threaded manner.
     *
     * Note:
     * All functions which access and modify the OSG tree must have the prefix osg and must be called from osgNodeCallback()!
     * Otherwise OSG could produce a segmentation fault.
     */
    class WLEMDDrawable: public boost::enable_shared_from_this< WLEMDDrawable >
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable > ConstSPtr;

        /**
         * Type for all scalars x, y, z. Similar to osg::value_type.
         */
        typedef float ValueT;

        /**
         * Class name for logs.
         */
        static const std::string CLASS;

        /**
         * Constructor
         *
         * @param widget widget to fill
         */
        explicit WLEMDDrawable( WCustomWidget::SPtr widget );

        /**
         * Destructor
         */
        virtual ~WLEMDDrawable();

        /**
         * Cast to DRAWABLE if possible.
         *
         * @return Shared Pointer< DRAWABLE >
         */
        template< typename DRAWABLE >
        boost::shared_ptr< DRAWABLE > getAs()
        {
            return boost::dynamic_pointer_cast< DRAWABLE >( shared_from_this() );
        }

        /**
         * Cast to DRAWABLE if possible.
         *
         * @return Shared Pointer< const DRAWABLE >
         */
        template< typename DRAWABLE >
        boost::shared_ptr< const DRAWABLE > getAs() const
        {
            return boost::dynamic_pointer_cast< DRAWABLE >( shared_from_this() );
        }

        /**
         * Invokes a draw with the new data.
         *
         * @param emm data to draw.
         */
        virtual void draw( LaBP::WLDataSetEMM::SPtr emm ) = 0;

        /**
         * Invokes a draw with the last data.
         */
        virtual void redraw();

        /**
         * Checks whether data is available.
         */
        virtual bool hasData() const = 0;

        /**
         * Gets modality to draw.
         *
         * @return modality
         */
        virtual LaBP::WEModalityType::Enum getModality() const;

        /**
         * Sets modality to draw.
         */
        virtual bool setModality( LaBP::WEModalityType::Enum modality );

        /**
         * Returns the widget to fill.
         *
         * @return widget
         */
        virtual WCustomWidget::SPtr getWidget() const;

        /**
         * Invokes a clear on the widget. Only The elements of this drawable should be removed.
         *
         * @param force
         */
        virtual void clearWidget( bool force = false ) = 0;

    protected:
        /**
         * Modifies and draws the scene graph.
         * This method is called as a callback function by the OSG system.
         * Modifications on the widget are allowed in this method (and sub calls) only! (Ensures single threaded access)
         */
        virtual void osgNodeCallback( osg::NodeVisitor* nv ) = 0;

        virtual bool mustDraw() const;

        virtual void resetDrawFlags();

        /**
         * Root node for all OSG objects which are added to the widget. osgNodeCallback() is applied to this node.
         */
        osg::ref_ptr< osg::Group > m_rootGroup;

        /**
         * Modality to draw.
         */
        LaBP::WEModalityType::Enum m_modality;

        /**
         * A flag to indicate a change of m_cb_emm.
         */
        bool m_dataChanged;

        /**
         * A flag to indicate a change of modality.
         */
        bool m_modalityChanged;

        /**
         * If we change the value of m_widget, than we will have a dangling event handler on the old widget. This can cause some problems.
         */
        const WCustomWidget::SPtr m_widget;

    private:
        /**
         * A flag to indicate osgNodeCallback() that the data should be draw.
         */
        bool m_draw;

        /**
         * A wrapper of WLEMDDrawable to register it as a callback for m_rootGroup.
         */
        class WLEMDDrawableCallbackDelegator: public osg::NodeCallback
        {
        public:
            /**
             * Constructor
             *
             * @param drawable Object to delegate the callback to.
             */
            WLEMDDrawableCallbackDelegator( WLEMDDrawable* drawable );

            /**
             * Destructor.
             */
            virtual ~WLEMDDrawableCallbackDelegator();

            /**
             * Calls the osgNodeCallback() of the wrapped WLEMDDrawable
             *
             * @param node
             * @param nv
             */
            void operator()( osg::Node* node, osg::NodeVisitor* nv );

        private:
            /**
             * WLEMDDrawable to wrap.
             */
            WLEMDDrawable* const m_drawable;
        };

        /**
         * Wrapper to register an instance of this class as a callback.
         */
        WLEMDDrawableCallbackDelegator* m_callbackDelegator;
    };
} /* namespace LaBP */
#endif  // WLEMDDRAWABLE_H_
