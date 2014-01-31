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

#include <string>
#include <utility>  // pair.make_pair
#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Geode>
#include <osg/ref_ptr>

#include <core/graphicsEngine/WGEGroupNode.h>
#include <core/gui/WCustomWidget.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"

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

        typedef osg::Vec4Array WLColorArray;

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
        static WLEMDDrawable2D::SPtr getInstance( WCustomWidget::SPtr widget, WLEModality::Enum modality, WEGraphType::Enum type =
                        WEGraphType::MULTI );

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
        virtual void draw( WLEMMeasurement::SPtr emm ) = 0;

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

        virtual float getSelectedTime() const;

        virtual bool setSelectedTime( float relative );

        /**
         * Gets a pair of EMM with the relative index for a x-coordinate.
         * Mapping between on screen data and internal data.
         *
         * \param pixel x-coordinate given in pixel.
         * \return A pair of EMM and the relative index for this block.
         *
         * \throws WOutOfBounds if pixel is out of view
         * \throws WLNoDataException if drawable has no data.
         */
        virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const = 0;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        virtual bool mustDraw() const;

        osg::ref_ptr< osg::Geode > drawChannel( const WLEMData::ChannelT& channel );

        void osgAddMarkLine();
        void osgAddTimeGrid();

        virtual size_t maxChannels( const WLEMData& emd ) const = 0;

        osg::ref_ptr< WGEGroupNode > m_channelGroup;
        osg::ref_ptr< osg::Geode > m_markerGeode;
        osg::ref_ptr< WGEGroupNode > m_timeGridGroup;
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

        osg::ref_ptr< WLColorArray > m_channelColors;
        osg::ref_ptr< WLColorArray > m_markerColors;
        osg::ref_ptr< WLColorArray > m_gridColors;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2D_H_
