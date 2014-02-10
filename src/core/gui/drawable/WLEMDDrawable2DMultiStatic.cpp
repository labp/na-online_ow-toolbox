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

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include <core/common/exceptions/WOutOfBounds.h>
#include <core/graphicsEngine/WGEGroupNode.h>

#include "core/exception/WLNoDataException.h"

#include "WLEMDDrawable2DMultiStatic.h"

namespace LaBP
{
    const std::string WLEMDDrawable2DMultiStatic::CLASS = "WLEMDDrawable2DMultiStatic";

    WLEMDDrawable2DMultiStatic::WLEMDDrawable2DMultiStatic( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2DMultiChannel( widget )
    {
        m_triggerColors = new WLColorArray;
        m_triggerColors->push_back( WColor( 0.5, 1.0, 0.5, 0.4 ) );

        osg::ref_ptr< osg::StateSet > state = m_rootGroup->getOrCreateStateSet();
        state->setMode( GL_BLEND, osg::StateAttribute::ON );
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
        if( !m_emm )
        {
            throw WLNoDataException( "No EMM available!" );
        }

        const ValueT width = m_widget->width();
        const WLEMMeasurement::SPtr emm = m_emm;
        if( m_xOffset <= pixel && pixel < width )
        {
            size_t sample = 0;
            const ValueT rel_width = width - m_xOffset;
            const ValueT rel_pix = pixel - m_xOffset;
            const size_t samples_total = emm->getModality( m_modality )->getSamplesPerChan();

            sample = rel_pix * samples_total / rel_width;

            wlog::debug( CLASS ) << "selected sample: " << sample;
            WAssertDebug( 0 <= sample && sample < samples_total, "0 <= sample && sample < samples_total" );
            return std::make_pair( emm, sample );
        }
        else
        {
            throw WOutOfBounds( "Pixel out of bounds!" );
        }
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
        osgSetTrigger( *( emm->getEventChannels() ) );

        WLEMDDrawable2DMultiChannel::osgNodeCallback( nv );
    }

    void WLEMDDrawable2DMultiStatic::osgAddChannels( const WLEMData& emd )
    {
        m_rootGroup->removeChild( m_channelGroup );
        m_channelGroup = new WGEGroupNode;

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

    void WLEMDDrawable2DMultiStatic::osgSetTrigger( const WLEMMeasurement::EDataT& events )
    {
        const ValueT pxWidth = static_cast< ValueT >( m_widget->width() - m_xOffset );

        // delete old trigger
        if( m_triggerGeode.valid() )
        {
            m_rootGroup->removeChild( m_triggerGeode );
            m_triggerGeode = NULL;
        }

        // find new trigger
        WLEMMeasurement::EDataT::const_iterator chanIt;
        for( chanIt = events.begin(); chanIt != events.end(); ++chanIt )
        {
            const WLEMMeasurement::EChannelT channel = *chanIt;
            for( size_t i = 0; i < channel.size(); ++i )
            {
                if( channel[i] > 0 )
                {
                    const size_t start = i;
                    do
                    {
                        ++i;
                    } while( channel[start] == channel[i] && i < channel.size() );

                    const ValueT pxStart = static_cast< ValueT >( start ) / static_cast< ValueT >( channel.size() ) * pxWidth
                                    + m_xOffset;
                    const ValueT pxEnd = static_cast< ValueT >( i - 1 ) / static_cast< ValueT >( channel.size() ) * pxWidth
                                    + m_xOffset;

                    // draw new trigger
                    osg::ref_ptr< osg::Geometry > geometry = new osg::Geometry;

                    osg::ref_ptr< osg::Vec2Array > vertices = new osg::Vec2Array();
                    vertices->reserve( 4 );
                    vertices->push_back( osg::Vec2( pxStart, 0.0 ) );
                    vertices->push_back( osg::Vec2( pxStart, m_widget->height() ) );
                    vertices->push_back( osg::Vec2( pxEnd, m_widget->height() ) );
                    vertices->push_back( osg::Vec2( pxEnd, 0.0 ) );

                    geometry->setVertexArray( vertices );
                    geometry->setColorArray( m_triggerColors );
                    geometry->setColorBinding( osg::Geometry::BIND_OVERALL );
                    geometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::QUADS, 0, vertices->size() ) );

                    if( !m_triggerGeode.valid() )
                    {
                        m_triggerGeode = new osg::Geode();
                    }
                    m_triggerGeode->addDrawable( geometry );
                }
            }
        }

        if( m_triggerGeode.valid() )
        {
            m_rootGroup->addChild( m_triggerGeode );
        }
    }

} /* namespace LaBP */
