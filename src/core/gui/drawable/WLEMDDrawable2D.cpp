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

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/PrimitiveSet>

#include <core/common/WAssert.h>
#include <core/gui/WCustomWidget.h>
#include <core/graphicsEngine/WGEGroupNode.h>

#include "core/data/emd/WLEMD.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WLEMDDrawable.h"
#include "WLEMDDrawable2D.h"
#include "WLEMDDrawable2DMultiStatic.h"
#include "WLEMDDrawable2DMultiStaticSource.h"
#include "WLEMDDrawable2DSingleChannel.h"
#include "WLEMDDrawable2DSingleSource.h"
#include "WLEMDDrawable2DMultiDynamic.h"
#include "WLEMDDrawable2DMultiDynamicSource.h"

namespace LaBP
{
    const string WLEMDDrawable2D::CLASS = "WLEMDDrawable2D";

    WLEMDDrawable2D::WLEMDDrawable2D( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable( widget )
    {
        m_xOffset = 0.0;
        m_yOffset = 32.0;
        m_timeRange = 0.0;
        m_timeRangeChanged = true;
        m_amplitudeScale = 1.5e-9;
        m_amplitudeScaleChanged = true;
        m_channelColors = new osg::Vec4Array;
        m_channelColors->push_back( osg::Vec4( 0.0, 0.0, 0.0, 1.0 ) );
        m_markerColors = new osg::Vec4Array;
        m_markerColors->push_back( osg::Vec4( 0.75f, 0.0f, 0.0f, 1.0f ) );
        m_selectedPixel = -1;
        m_selectedPixelChanged = false;
    }

    WLEMDDrawable2D::~WLEMDDrawable2D()
    {
    }

    osg::ref_ptr< osg::Geode > WLEMDDrawable2D::drawChannel( const LaBP::WLEMD::ChannelT& channel )
    {
        osg::ref_ptr< osg::DrawArrays > lineDrawer = new osg::DrawArrays( osg::PrimitiveSet::LINE_STRIP, 0, channel.size() );

        osg::ref_ptr< osg::Vec2Array > samplesDots = new osg::Vec2Array();
        const size_t samples_begin = 0;
        const size_t samples_end = channel.size();
        for( std::size_t sample = samples_begin; sample < samples_end; ++sample )
        {
            samplesDots->push_back( osg::Vec2( sample, channel[sample] ) );
        }

        // Create geometry to draw 2D Points
        osg::ref_ptr< osg::Geometry > packetGeometry = new osg::Geometry;
        packetGeometry->setVertexArray( samplesDots );
        packetGeometry->setColorArray( m_channelColors );
        packetGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
        packetGeometry->setDataVariance( osg::Object::DYNAMIC );
        packetGeometry->addPrimitiveSet( lineDrawer );

        // Geode to add drawable to scene graph
        osg::ref_ptr< osg::Geode > channelGeode = new osg::Geode;
        channelGeode->addDrawable( packetGeometry );
        return channelGeode;
    }

    void WLEMDDrawable2D::osgAddMarkLine()
    {
        if( m_selectedPixel > 0 )
        {
            m_rootGroup->removeChild( m_markerGeode );
            m_markerGeode = new osg::Geode;

            osg::ref_ptr< osg::Geometry > geometry = new osg::Geometry;

            osg::ref_ptr< osg::Vec2Array > vertices = new osg::Vec2Array();
            vertices->reserve( 2 );
            vertices->push_back( osg::Vec2( m_selectedPixel, 0.0f ) );
            vertices->push_back( osg::Vec2( m_selectedPixel, m_widget->height() ) );

            geometry->setVertexArray( vertices );
            geometry->setColorArray( m_markerColors );
            geometry->setColorBinding( osg::Geometry::BIND_OVERALL );
            geometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, 2 ) );

            m_markerGeode->addDrawable( geometry );
            m_rootGroup->addChild( m_markerGeode );
        }
    }

    WLEMDDrawable::ValueT WLEMDDrawable2D::getTimeRange() const
    {
        return m_timeRange;
    }

    bool WLEMDDrawable2D::setTimeRange( ValueT timeRange )
    {
        if( timeRange != m_timeRange )
        {
            m_timeRange = timeRange;
            m_timeRangeChanged = true;
            return true;
        }
        return false;
    }

    WLEMDDrawable::ValueT WLEMDDrawable2D::getAmplitudeScale() const
    {
        return m_amplitudeScale;
    }

    bool WLEMDDrawable2D::setAmplitudeScale( ValueT value )
    {
        if( value != m_amplitudeScale )
        {
            m_amplitudeScale = value;
            m_amplitudeScaleChanged = true;
            return true;
        }
        return false;
    }

    WLEMDDrawable::ValueT WLEMDDrawable2D::getSelectedPixel() const
    {
        return m_selectedPixel;
    }

    bool WLEMDDrawable2D::setSelectedPixel( ValueT value )
    {
        if( value != m_selectedPixel )
        {
            m_selectedPixel = value;
            m_selectedPixelChanged = true;
            return true;
        }
        return false;
    }

    WLEMDDrawable2D::SPtr WLEMDDrawable2D::getInstance( WCustomWidget::SPtr widget, LaBP::WEModalityType::Enum modality,
                    WEGraphType::Enum type )
    {
        WLEMDDrawable2D::SPtr drawable2D;
        switch( type )
        {
            case WEGraphType::MULTI:
                if( modality == LaBP::WEModalityType::SOURCE )
                {
                    drawable2D.reset( new WLEMDDrawable2DMultiStaticSource( widget ) );
                }
                else
                {
                    drawable2D.reset( new WLEMDDrawable2DMultiStatic( widget ) );
                }
                break;
            case WEGraphType::SINGLE:
                if( modality == LaBP::WEModalityType::SOURCE )
                {
                    drawable2D.reset( new WLEMDDrawable2DSingleSource( widget ) );
                }
                else
                {
                    drawable2D.reset( new WLEMDDrawable2DSingleChannel( widget ) );
                }
                break;
            case WEGraphType::DYNAMIC:
                if( modality == LaBP::WEModalityType::SOURCE )
                {
                    drawable2D.reset( new WLEMDDrawable2DMultiDynamicSource( widget ) );
                }
                else
                {
                    drawable2D.reset( new WLEMDDrawable2DMultiDynamic( widget ) );
                }
                break;
            default:
                WAssert( false, "Unknown WEGraphType!" );
                break;
        }
        drawable2D->setModality( modality );
        return drawable2D;
    }

    void WLEMDDrawable2D::clearWidget( bool force )
    {
        // TODO(pieloth) insert code to delete nodes from widget here!
    }

} /* namespace LaBP */
