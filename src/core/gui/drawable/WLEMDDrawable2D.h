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

#ifndef WLEMDDRAWABLE2D_H_
#define WLEMDDRAWABLE2D_H_

#include <string>
#include <utility>  // pair.make_pair
#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Geode>
#include <osg/ref_ptr>

#include <core/graphicsEngine/WGEGroupNode.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"

#include "WLEMDDrawable.h"

/**
 * Abstract class to draw 2D graph like visualization of EMD data.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2D: public WLEMDDrawable
{
public:
    typedef boost::shared_ptr< WLEMDDrawable2D > SPtr; //!< Abbreviation for a shared pointer on a instance of this class.

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2D > ConstSPtr;

    typedef osg::Vec4Array WLColorArray;

    static const std::string CLASS; //!< Class name for logging purpose.

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
     * \param widget widget to fill.
     * \param modality modality to draw.
     * \param type type of visualization.
     *
     * \return Instance of a WLEMDDrawable2D implementation.
     */
    static WLEMDDrawable2D::SPtr getInstance( WUIViewWidget::SPtr widget, WLEModality::Enum modality, WEGraphType::Enum type =
                    WEGraphType::MULTI );

    /**
     * Constructor.
     *
     * \param widget
     */
    explicit WLEMDDrawable2D( WUIViewWidget::SPtr widget );

    /**
     * Destructor.
     */
    virtual ~WLEMDDrawable2D();

    virtual void draw( WLEMMeasurement::SPtr emm ) = 0;

    bool hasData() const = 0;

    /**
     * Gets the time range to visualize. TODO(pieloth): seconds or millimeter?
     *
     * \return Time range in ???.
     */
    virtual ValueT getTimeRange() const;

    /**
     * Sets the time range to visualize. TODO(pieloth): seconds or millimeter?
     *
     * \param timeRange Time range in ???
     * \return True if time range has changed.
     */
    virtual bool setTimeRange( ValueT timeRange );

    /**
     * Gets the amplitude scale (max. value).
     *
     * \return The amplitude scale.
     */
    virtual ValueT getAmplitudeScale() const;

    /**
     * Sets the amplitude scale.
     * \param value Maximum value.
     * \return True if scale has changed.
     */
    virtual bool setAmplitudeScale( ValueT value );

    /**
     * Gets the selected pixel in the drawing area.
     *
     * \return  The selected pixel in the drawing area.
     */
    virtual ValueT getSelectedPixel() const;

    /**
     * Sets the selected pixel relative to the drawing area.
     *
     * \param value Selected pixel.
     * \return True if selected pixel has changed.
     */
    virtual bool setSelectedPixel( ValueT value );

    /**
     * Gets the selected point in time. TODO(pieloth): seconds or milliseconds?
     *
     * \return The selected point in time in ???.
     */
    virtual float getSelectedTime() const;

    /**
     * Sets the selected point in time relative to the widget width. TODO(pieloth): seconds or milliseconds?
     *
     * \param relative Point in time in ???.
     * \return Selected point in time.
     */
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

    /**
     * Draws the graph of channel data.
     *
     * \param channel Channel to draw.
     * \return OSG Geode containing graphical representation.
     */
    osg::ref_ptr< osg::Geode > drawChannel( const WLEMData::ChannelT& channel );

    /**
     * Draws and adds the mark line for the time selection.
     */
    void osgAddMarkLine();

    /**
     * Draws and adds the time grid.
     */
    void osgAddTimeGrid();

    /**
     * Calculates the number of channels which can be drawn on screen.
     * It depends on the graph type, channel count, view height and so on.
     *
     * \param emd Data to draw.
     * \return Maximum number channels to draw.
     */
    virtual size_t maxChannels( const WLEMData& emd ) const = 0;

    osg::ref_ptr< WGEGroupNode > m_channelGroup; //!< Contains a graphical object for each channel.
    osg::ref_ptr< osg::Geode > m_markerGeode; //!< Contains the marker line.
    osg::ref_ptr< WGEGroupNode > m_timeGridGroup; //!< Contains the time grid.
    ValueT m_timeGridWidth;
    ValueT m_timeGridHight;

    ValueT m_timeRange;
    bool m_timeRangeChanged;

    ValueT m_xOffset; //!< Time-axis offset for raw data visualization, e.g. for labels.

    ValueT m_yOffset; //!< Channel or value offset for raw data visualization, e.g. for time scale.

    ValueT m_amplitudeScale;
    bool m_amplitudeScaleChanged;

    ValueT m_selectedPixel;
    bool m_selectedPixelChanged;

    osg::ref_ptr< WLColorArray > m_channelColors;
    osg::ref_ptr< WLColorArray > m_markerColors;
    osg::ref_ptr< WLColorArray > m_gridColors;
};

#endif  // WLEMDDRAWABLE2D_H_
