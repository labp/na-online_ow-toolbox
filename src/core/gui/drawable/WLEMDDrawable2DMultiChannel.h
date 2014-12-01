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

#ifndef WLEMDDRAWABLE2DMULTICHANNEL_H_
#define WLEMDDRAWABLE2DMULTICHANNEL_H_

#include <string>
#include <utility>  // for pair<>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>

#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMData.h"

#include "WLEMDDrawable2D.h"

/**
 * Template for a multi-channel 2D view.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2DMultiChannel: public WLEMDDrawable2D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable2DMultiChannel > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2DMultiChannel > ConstSPtr;

    static const std::string CLASS; //!< Class name for logging purpose.

    explicit WLEMDDrawable2DMultiChannel( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable2DMultiChannel();

    virtual void draw( WLEMMeasurement::SPtr emm ) = 0;

    virtual bool hasData() const = 0;

    /**
     * Gets the height or spacing for one channel.
     *
     * \return The height for one channel.
     */
    virtual ValueT getChannelHeight() const;

    /**
     * Sets the height or spacing for one channel.
     *
     * \param spacing The height or spacing for one channel.
     */
    virtual void setChannelHeight( ValueT spacing );

    /**
     * Creates a pair of the selected point and the corresponding data sample.
     *
     * \param pixel Selected pixel.
     * \return A pair of a data sample and a selected pixel.
     */
    virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const = 0;

    /**
     * Gets the first channel which is visualized.
     *
     * \return First channel to visualize.
     */
    virtual size_t getChannelBegin() const;

    /**
     * Sets the first channel to visualized. This is used for scrolling over the channels.
     *
     * \param channelNr Channel number.
     * \return The channel number is used.
     */
    virtual size_t setChannelBegin( size_t channelNr );

protected:
    virtual bool mustDraw() const;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    /**
     * Draws and adds the labels for each channel.
     *
     * \param emd EM data containing the channel names.
     */
    void virtual osgAddLabels( const WLEMData& emd );

    virtual size_t maxChannels( const WLEMData& emd ) const;

    /**
     * Calculates the first channel to draw considering the available data.
     * \param emd Data t draw.
     * \return Channel index.
     */
    virtual size_t getChannelBegin( const WLEMData& emd );

    const ValueT m_labelWidth;

    ValueT m_channelHeight;
    bool m_channelHeightChanged;

    size_t m_channelBegin;
    bool m_channelBeginChanged;

    osg::ref_ptr< osg::PositionAttitudeTransform > m_labelsText; //!< Contains the channel labels.
    osg::ref_ptr< osg::Geode > m_labelsBackground; //!< Contains the intransparent labels background.
};

#endif  // WLEMDDRAWABLE2DMULTICHANNEL_H_
