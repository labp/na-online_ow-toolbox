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

#include <cmath>    // fabs
#include <string>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/PrimitiveSet>
#include <osgText/Text>

#include <core/common/WAssert.h>
#include <core/common/WColor.h>
#include <core/gui/WCustomWidget.h>
#include <core/graphicsEngine/WGEGroupNode.h>

#include "core/data/emd/WLEMData.h"
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
        m_yOffset = 0.0;
        m_timeRange = 0.0;
        m_timeRangeChanged = true;
        m_amplitudeScale = 1.5e-9;
        m_amplitudeScaleChanged = true;
        m_channelColors = new WLColorArray;
        m_channelColors->push_back( defaultColor::BLACK );
        m_markerColors = new WLColorArray;
        m_markerColors->push_back( defaultColor::DARKRED );
        m_gridColors = new WLColorArray;
        m_gridColors->push_back( defaultColor::ORANGE );
        m_selectedPixel = -1;
        m_selectedPixelChanged = false;
        m_timeGridWidth = -1.0f;
        m_timeGridHight = -1.0f;

        osg::ref_ptr< osg::StateSet > state = m_rootGroup->getOrCreateStateSet();
        state->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
        state->setMode( GL_DEPTH_TEST, osg::StateAttribute::OFF );
    }

    WLEMDDrawable2D::~WLEMDDrawable2D()
    {
    }

    bool WLEMDDrawable2D::mustDraw() const
    {
        return WLEMDDrawable::mustDraw() || m_timeRangeChanged || m_amplitudeScaleChanged || m_selectedPixelChanged;
    }

    osg::ref_ptr< osg::Geode > WLEMDDrawable2D::drawChannel( const WLEMData::ChannelT& channel )
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

    void WLEMDDrawable2D::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( mustDraw() )
        {
            osgAddMarkLine();
            osgAddTimeGrid();
        }

        m_timeRangeChanged = false;
        m_amplitudeScaleChanged = false;
        m_selectedPixelChanged = false;

        WLEMDDrawable::resetDrawFlags();
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
            vertices->push_back( osg::Vec2( m_selectedPixel - 1, 0.0f ) );
            vertices->push_back( osg::Vec2( m_selectedPixel - 1, m_widget->height() ) );
            vertices->push_back( osg::Vec2( m_selectedPixel, 0.0f ) );
            vertices->push_back( osg::Vec2( m_selectedPixel, m_widget->height() ) );
            vertices->push_back( osg::Vec2( m_selectedPixel + 1, 0.0f ) );
            vertices->push_back( osg::Vec2( m_selectedPixel + 1, m_widget->height() ) );

            geometry->setVertexArray( vertices );
            geometry->setColorArray( m_markerColors );
            geometry->setColorBinding( osg::Geometry::BIND_OVERALL );
            geometry->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, vertices->size() ) );

            m_markerGeode->addDrawable( geometry );
            m_rootGroup->addChild( m_markerGeode );
        }
    }

    void WLEMDDrawable2D::osgAddTimeGrid()
    {
        const ValueT height = m_widget->height();
        const ValueT width = m_widget->width();
        if( m_timeRangeChanged || m_timeGridWidth != width || m_timeGridHight != height )
        {
            m_timeGridWidth = width;
            m_timeGridHight = height;

            m_rootGroup->removeChild( m_timeGridGroup );
            m_timeGridGroup = new osg::Group;

            const ValueT pxPerSec = ( width - m_xOffset ) / getTimeRange();
            const ValueT deltaT = pxPerSec * 0.05;

            for( ValueT xPos = m_xOffset; xPos < width; xPos += deltaT )
            {
                osg::ref_ptr< osg::Geometry > line = new osg::Geometry;

                osg::ref_ptr< osg::Vec2Array > vertices = new osg::Vec2Array();
                vertices->reserve( 2 );
                vertices->push_back( osg::Vec2( xPos, 0.0f ) );
                vertices->push_back( osg::Vec2( xPos, height ) );

                line->setVertexArray( vertices );
                line->setColorArray( m_gridColors );
                line->setColorBinding( osg::Geometry::BIND_OVERALL );
                line->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, vertices->size() ) );

                osg::ref_ptr< osg::Geode > lineGeode = new osg::Geode;
                lineGeode->addDrawable( line );
                m_timeGridGroup->addChild( lineGeode );
            }

            // Because GL_DEPTH_TEST is OFF this, order is important:
            // 1. graph data
            // 2. label background
            // 3. labes text
            //
            // NOTE: Solve alpha/transparent/opaque problem to use z coordinate for hiding objects!

            // Create background rectangle for labels //
            osg::ref_ptr< osg::Vec3Array > bgVertices = new osg::Vec3Array;
            const ValueT z = -1.0f;
            bgVertices->push_back( osg::Vec3( m_xOffset, height - 16, z ) );
            bgVertices->push_back( osg::Vec3( m_xOffset, height, z ) );
            bgVertices->push_back( osg::Vec3( m_xOffset + 95, height, z ) );
            bgVertices->push_back( osg::Vec3( m_xOffset + 95, height - 16, z ) );

            osg::ref_ptr< WLColorArray > bgColors = new WLColorArray;
            bgColors->push_back( defaultColor::WHITE );

            osg::ref_ptr< osg::Geometry > background = new osg::Geometry;
            background->setVertexArray( bgVertices );
            background->setColorArray( bgColors );
            background->setColorBinding( osg::Geometry::BIND_OVERALL );
            background->addPrimitiveSet( new osg::DrawArrays( GL_QUADS, 0, 4 ) );
            osg::ref_ptr< osg::Geode > backgroundGeode = new osg::Geode;
            backgroundGeode->addDrawable( background );
            m_timeGridGroup->addChild( backgroundGeode );

            // Create time scale text.
            osg::ref_ptr< osgText::Text > text = new osgText::Text;
            text->setText( "50ms/DIV" ); // related to deltaT
            text->setPosition( osg::Vec3( m_xOffset, height, 0.0 ) );
            text->setAlignment( osgText::Text::LEFT_TOP );
            text->setAxisAlignment( osgText::Text::SCREEN );
            text->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
            text->setCharacterSize( 16 );
            text->setColor( ( *m_gridColors )[0] );
            osg::ref_ptr< osg::Geode > textGeode = new osg::Geode;
            textGeode->addDrawable( text );
            m_timeGridGroup->addChild( textGeode );

            m_rootGroup->addChild( m_timeGridGroup );
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
        if( fabs( value - m_selectedPixel ) >= 1.0 )
        {
            m_selectedPixel = value;
            m_selectedPixelChanged = true;
            return true;
        }
        return false;
    }

    float WLEMDDrawable2D::getSelectedTime() const
    {
        return ( m_selectedPixel - m_xOffset ) / ( m_widget->width() - m_xOffset );
    }

    bool WLEMDDrawable2D::setSelectedTime( float relative )
    {
        if( relative < 0 )
        {
            return false;
        }
        const float pos = relative * ( m_widget->width() - m_xOffset ) + m_xOffset;
        return setSelectedPixel( pos );
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

} /* namespace LaBP */
