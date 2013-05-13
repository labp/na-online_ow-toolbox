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

#ifndef WLEMDDRAWABLE2D_H_
#define WLEMDDRAWABLE2D_H_

#include <cstddef>
#include <string>

#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Group>
#include <osg/ref_ptr>

#include <core/gui/WCustomWidget.h>

#include "core/util/WLRingBuffer.h"
#include "core/data/emd/WLEMD.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WLEMDDrawable.h"

namespace LaBP
{
    /**
     * Abstract class to draw 2D graph like visualization of EMD data.
     */
    class WLEMDDrawable2D: public WLEMDDrawable
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable2D > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable2D > ConstSPtr;

        /**
         * Class name for logs.
         */
        static const std::string CLASS;

        /**
         * Enumeration to select a single/overlay or multi channel graph.
         */
        struct WEGraphType
        {
            enum Enum
            {
                SINGLE, MULTI, DYNAMIC
            };
        };

        /**
         * Creates an instance for the requested parameters.
         *
         * @param widget widget to fill.
         * @param modality modality to draw.
         * @param type type of visualization.
         *
         * @return Instance of a WLEMDDrawable2D implementation.
         */
        static WLEMDDrawable2D::SPtr getInstance( WCustomWidget::SPtr widget, LaBP::WEModalityType::Enum modality,
                        WEGraphType::Enum type = WEGraphType::MULTI );

        /**
         * Constructor.
         *
         * @param widget
         */
        explicit WLEMDDrawable2D( WCustomWidget::SPtr widget );

        /**
         * Destructor.
         */
        virtual ~WLEMDDrawable2D();

        /**
         * Invokes a draw with the new data.
         *
         * @param emm data to draw.
         */
        virtual void draw( LaBP::WLDataSetEMM::SPtr emm ) = 0;

        virtual void redraw() = 0;

        /**
         * Checks whether data is available.
         */
        bool hasData() const = 0;

        virtual ValueT getTimeRange() const;
        virtual bool setTimeRange( ValueT timeRange );

        virtual ValueT getAmplitudeScale() const;
        virtual bool setAmplitudeScale( ValueT value );

        virtual ValueT getSelectedPixel() const;
        virtual bool setSelectedPixel( ValueT value );

        virtual void clearWidget( bool force = false );

        virtual std::pair< LaBP::WLDataSetEMM::SPtr, size_t > getSelectedData( ValueT pixel ) const = 0;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        osg::ref_ptr< osg::Geode > drawChannel( const LaBP::WLEMD::ChannelT& channel );

        void osgAddMarkLine();
        void osgAddTimeGrid();

        virtual size_t maxChannels( const LaBP::WLEMD* emd ) const = 0;

        osg::ref_ptr< osg::Group > m_channelGroup;
        osg::ref_ptr< osg::Geode > m_markerGeode;
        osg::ref_ptr< osg::Group > m_timeGridGroup;
        ValueT m_timeGridWidth;
        ValueT m_timeGridHight;

        ValueT m_timeRange;
        bool m_timeRangeChanged;

        /**
         * Time-axis offset for raw data visualization, e.g. for labels
         */
        ValueT m_xOffset;

        /**
         * Channel or value offset for raw data visualization, e.g. for time scale
         */
        ValueT m_yOffset;

        ValueT m_amplitudeScale;
        bool m_amplitudeScaleChanged;

        ValueT m_selectedPixel;
        bool m_selectedPixelChanged;

    private:
        osg::ref_ptr< osg::Vec4Array > m_channelColors;
        osg::ref_ptr< osg::Vec4Array > m_markerColors;
        osg::ref_ptr< osg::Vec4Array > m_timeGridColors;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2D_H_
