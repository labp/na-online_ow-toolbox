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

#include <string>
#include <utility>  // for pair<>
#include <osg/MatrixTransform>

#include <core/common/WLogger.h>

#include "WLEMDDrawable2DMultiStatic.h"

namespace LaBP
{
    const std::string WLEMDDrawable2DMultiStatic::CLASS = "WLEMDDrawable2DMultiStatic";

    WLEMDDrawable2DMultiStatic::WLEMDDrawable2DMultiStatic( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2DMultiChannel( widget )
    {
    }

    WLEMDDrawable2DMultiStatic::~WLEMDDrawable2DMultiStatic()
    {
    }

    void WLEMDDrawable2DMultiStatic::draw( WLEMMeasurement::SPtr emm )
    {
        m_emm = emm;
        m_dataChanged = true;
        redraw();
    }

    bool WLEMDDrawable2DMultiStatic::hasData() const
    {
        return m_emm.get() && m_emm->hasModality( m_modality );
    }

    std::pair< WLEMMeasurement::SPtr, size_t > WLEMDDrawable2DMultiStatic::getSelectedData( ValueT pixel ) const
    {
        size_t sample = 0;
        const ValueT x_offset = m_xOffset;

        WLEMMeasurement::SPtr emm = m_emm;

        if( pixel > x_offset )
        {
            const ValueT width = m_widget->width() - x_offset;
            const ValueT relPix = pixel - x_offset;
            const size_t samples_total = emm->getModality( m_modality )->getSamplesPerChan();

            sample = relPix * samples_total / width;
        }
        wlog::debug( CLASS ) << "selected sample: " << sample;
        return std::make_pair( emm, sample );
    }

    void WLEMDDrawable2DMultiStatic::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( !mustDraw() )
            return;
        if( !hasData() )
        {
            return;
        }

        WLEMMeasurement::ConstSPtr emm = m_emm;
        WLEMData::ConstSPtr emd = emm->getModality( m_modality );
        osgAddLabels( *emd );
        osgAddChannels( *emd );

        WLEMDDrawable2DMultiChannel::osgNodeCallback( nv );
    }

    void WLEMDDrawable2DMultiStatic::osgAddChannels( const WLEMData& emd )
    {
        m_rootGroup->removeChild( m_channelGroup );
        m_channelGroup = new osg::Group;

        const ValueT x_pos = m_xOffset;
        const ValueT y_pos = m_widget->height() - m_yOffset;
        const ValueT width = m_widget->width() - x_pos;
        const ValueT x_scale = width / emd.getSamplesPerChan();
        const ValueT y_scale = ( m_channelHeight / 2 ) / m_amplitudeScale;

        osg::ref_ptr< osg::MatrixTransform > panTransform = new osg::MatrixTransform;
        panTransform->setDataVariance( osg::Object::DYNAMIC );
        // TODO(pieloth): dynamic shift scale ... width / m_timeRange
        // TODO(pieloth): dynamic shift scale ... x_pos * width / m_timeRange,
        panTransform->setMatrix( osg::Matrix::translate( x_pos, y_pos, 0.0 ) );
        const WLEMData::DataT& emdData = emd.getData();
        const size_t channels_emd = emd.getNrChans();
        const size_t channels_count = maxChannels( emd );
        osg::ref_ptr< osg::Geode > channelGeode;
        for( size_t channel = getChannelBegin( emd ), channelPos = 0; channelPos < channels_count && channel < channels_emd;
                        ++channel, ++channelPos )
        {
            channelGeode = drawChannel( emdData.row( channel ) );
            osg::ref_ptr< osg::MatrixTransform > scaleSpacingTransform = new osg::MatrixTransform;
            scaleSpacingTransform->setMatrix(
                            osg::Matrix::scale( x_scale, y_scale, 1.0 )
                                            * osg::Matrix::translate( 0.0, -m_channelHeight * channelPos, 0.0 ) );
            scaleSpacingTransform->setDataVariance( osg::Object::DYNAMIC );
            scaleSpacingTransform->addChild( channelGeode );

            panTransform->addChild( scaleSpacingTransform );
        }

        m_channelGroup->addChild( panTransform );
        m_rootGroup->addChild( m_channelGroup );
    }

} /* namespace LaBP */
