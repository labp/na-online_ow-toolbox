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

#include <sstream>
#include <string>
#include <utility>  // for pair<>
#include <boost/lexical_cast.hpp>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osgText/Text>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include <core/common/exceptions/WOutOfBounds.h>
#include <core/ui/WUIViewWidget.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/exception/WLNoDataException.h"
#include "core/util/bounds/WLBoundCalculator.h"
#include "core/util/strategy/WLParameter.h"

#include "WLEMDDrawable2DSingleChannel.h"

const std::string WLEMDDrawable2DSingleChannel::CLASS = "WLEMDDrawable2DSingleChannel";

WLEMDDrawable2DSingleChannel::WLEMDDrawable2DSingleChannel( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable2D( widget )
{
    m_valueGridWidth = -1;
    m_valueGridHeight = -1;

    m_triggerColors = new WLColorArray;
    m_triggerColors->push_back( WColor( 0.5, 1.0, 0.5, 0.4 ) );

    osg::ref_ptr< osg::StateSet > state = m_rootGroup->getOrCreateStateSet();
    state->setMode( GL_BLEND, osg::StateAttribute::ON );
}

WLEMDDrawable2DSingleChannel::~WLEMDDrawable2DSingleChannel()
{
    if( m_valueGridGroup.valid() )
    {
        m_rootGroup->remove( m_valueGridGroup );
        m_valueGridGroup = NULL;
    }
}

void WLEMDDrawable2DSingleChannel::draw( WLEMMeasurement::SPtr emm )
{
    m_emm = emm;
    m_dataChanged = true;
    redraw();
}

bool WLEMDDrawable2DSingleChannel::hasData() const
{
    return m_emm.get() && m_emm->hasModality( m_modality );
}

size_t WLEMDDrawable2DSingleChannel::maxChannels( const WLEMData& emd ) const
{
    return emd.getNrChans();
}

void WLEMDDrawable2DSingleChannel::osgAddChannels( const WLEMData& emd )
{
    m_rootGroup->removeChild( m_channelGroup );
    m_channelGroup = new WGEGroupNode;

    const ValueT x_pos = m_xOffset;
    const ValueT y_pos = m_widget->height() / 2;
    const ValueT width = m_widget->width() - x_pos;
    const ValueT x_scale = width / emd.getSamplesPerChan();
    const ValueT y_scale = ( m_widget->height() / 2 ) / m_amplitudeScale;

    osg::ref_ptr< osg::MatrixTransform > panTransform = new osg::MatrixTransform;
    panTransform->setDataVariance( osg::Object::DYNAMIC );
    // TODO(pieloth): dynamic shift scale ... width / m_timeRange
    // TODO(pieloth): dynamic shift scale ... x_pos * width / m_timeRange,
    panTransform->setMatrix( osg::Matrix::translate( x_pos, y_pos, 0.0 ) );
    const WLEMData::DataT& emdData = emd.getData();
    const size_t channels_begin = 0;
    const size_t channels_count = maxChannels( emd );
    osg::ref_ptr< osg::Geode > channelGeode;
    for( size_t channel = channels_begin, channelPos = 0; channelPos < channels_count && channel < emd.getNrChans();
                    ++channel, ++channelPos )
    {
        channelGeode = drawChannel( emdData.row( channel ) );
        osg::ref_ptr< osg::MatrixTransform > scaleSpacingTransform = new osg::MatrixTransform;
        scaleSpacingTransform->setMatrix( osg::Matrix::scale( x_scale, y_scale, 1.0 ) );
        scaleSpacingTransform->setDataVariance( osg::Object::DYNAMIC );
        scaleSpacingTransform->addChild( channelGeode );

        panTransform->addChild( scaleSpacingTransform );
    }

    m_channelGroup->addChild( panTransform );
    m_rootGroup->addChild( m_channelGroup );
}

void WLEMDDrawable2DSingleChannel::osgAddValueGrid( const WLEMData& emd )
{
    const ValueT height = m_widget->height();
    const ValueT width = m_widget->width();
    if( m_amplitudeScaleChanged || m_valueGridWidth != width || m_valueGridHeight != height )
    {
        m_valueGridWidth = width;
        m_valueGridHeight = height;

        m_rootGroup->removeChild( m_valueGridGroup );
        m_valueGridGroup = new WGEGroupNode;

        // Draw 0 line and add text
        osg::ref_ptr< osg::Geometry > line = new osg::Geometry;

        osg::ref_ptr< osg::Vec2Array > vertices = new osg::Vec2Array();
        vertices->reserve( 2 );
        const ValueT y_zero_pos = height / 2;
        vertices->push_back( osg::Vec2( m_xOffset, y_zero_pos ) );
        vertices->push_back( osg::Vec2( width, y_zero_pos ) );

        line->setVertexArray( vertices );
        line->setColorArray( m_gridColors ); // TODO(pieloth): define color for grid
        line->setColorBinding( osg::Geometry::BIND_OVERALL );
        line->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, vertices->size() ) );

        osg::ref_ptr< osg::Geode > lineGeode = new osg::Geode;
        lineGeode->addDrawable( line );
        m_valueGridGroup->addChild( lineGeode );

        const float char_size = 16;

        osg::ref_ptr< osgText::Text > zeroText = new osgText::Text;
        zeroText->setText( "0.0" ); // related to deltaT
        zeroText->setPosition( osg::Vec3( 0, y_zero_pos, 0.0 ) );
        zeroText->setAlignment( osgText::Text::LEFT_BOTTOM_BASE_LINE );
        zeroText->setAxisAlignment( osgText::Text::SCREEN );
        zeroText->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
        zeroText->setCharacterSize( char_size );
        zeroText->setColor( ( *m_gridColors )[0] );
        osg::ref_ptr< osg::Geode > zeroTextGeode = new osg::Geode;
        zeroTextGeode->addDrawable( zeroText );
        m_valueGridGroup->addChild( zeroTextGeode );

        // Find maximum
        WLBoundCalculator bc;
        WLEMData::ScalarT max = bc.getMax( emd.getData() );

        const ValueT y_scale = ( ( m_widget->height() / 2 ) / m_amplitudeScale );
        for( ValueT yPos = y_zero_pos + ( y_scale * max ); yPos > height * 0.9; yPos = y_zero_pos + ( y_scale * max ) )
        {
            max /= 2;
        }

        // Draw max and add text
        ValueT yPos = y_zero_pos + ( y_scale * max );
        osg::ref_ptr< osg::Vec2Array > vLinePosVec = new osg::Vec2Array();
        vLinePosVec->reserve( 2 );
        vLinePosVec->push_back( osg::Vec2( m_xOffset, yPos ) );
        vLinePosVec->push_back( osg::Vec2( width, yPos ) );

        osg::ref_ptr< osg::Geometry > vLinePos = new osg::Geometry;
        vLinePos->setVertexArray( vLinePosVec );
        vLinePos->setColorArray( m_gridColors );
        vLinePos->setColorBinding( osg::Geometry::BIND_OVERALL );
        vLinePos->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, vLinePosVec->size() ) );

        osg::ref_ptr< osgText::Text > maxText = new osgText::Text;
        std::stringstream ssMax;
        ssMax << std::setprecision( 2 ) << std::scientific << max;
        maxText->setText( ssMax.str() );
        maxText->setPosition( osg::Vec3( 0, yPos, 0.0 ) );
        maxText->setAlignment( osgText::Text::LEFT_BOTTOM_BASE_LINE );
        maxText->setAxisAlignment( osgText::Text::SCREEN );
        maxText->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
        maxText->setCharacterSize( char_size );
        maxText->setColor( ( *m_gridColors )[0] );
        osg::ref_ptr< osg::Geode > maxTextGeode = new osg::Geode;
        zeroTextGeode->addDrawable( maxText );
        m_valueGridGroup->addChild( maxTextGeode );

        // Draw min and add text
        yPos = y_zero_pos - ( y_scale * max );
        osg::ref_ptr< osg::Vec2Array > vLineNegVec = new osg::Vec2Array();
        vLineNegVec->reserve( 2 );
        vLineNegVec->push_back( osg::Vec2( m_xOffset, yPos ) );
        vLineNegVec->push_back( osg::Vec2( width, yPos ) );

        osg::ref_ptr< osg::Geometry > vLineNeg = new osg::Geometry;
        vLineNeg->setVertexArray( vLineNegVec );
        vLineNeg->setColorArray( m_gridColors );
        vLineNeg->setColorBinding( osg::Geometry::BIND_OVERALL );
        vLineNeg->addPrimitiveSet( new osg::DrawArrays( osg::PrimitiveSet::LINES, 0, vLineNegVec->size() ) );

        osg::ref_ptr< osgText::Text > minText = new osgText::Text;
        std::stringstream ssMin;
        ssMin << std::setprecision( 2 ) << std::scientific << -max;
        minText->setText( ssMin.str() );
        minText->setPosition( osg::Vec3( 0, yPos, 0.0 ) );
        minText->setAlignment( osgText::Text::LEFT_BOTTOM_BASE_LINE );
        minText->setAxisAlignment( osgText::Text::SCREEN );
        minText->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
        minText->setCharacterSize( char_size );
        minText->setColor( ( *m_gridColors )[0] );
        osg::ref_ptr< osg::Geode > minTextGeode = new osg::Geode;
        zeroTextGeode->addDrawable( minText );
        m_valueGridGroup->addChild( minTextGeode );

        osg::ref_ptr< osg::Geode > valueGeode = new osg::Geode;
        valueGeode->addDrawable( vLinePos );
        valueGeode->addDrawable( vLineNeg );
        m_valueGridGroup->addChild( valueGeode );

        m_rootGroup->addChild( m_valueGridGroup );
    }
}

std::pair< WLEMMeasurement::SPtr, size_t > WLEMDDrawable2DSingleChannel::getSelectedData( ValueT pixel ) const
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

void WLEMDDrawable2DSingleChannel::osgNodeCallback( osg::NodeVisitor* nv )
{
    if( !mustDraw() )
    {
        return;
    }
    if( !hasData() )
    {
        return;
    }

    WLEMMeasurement::ConstSPtr emm = m_emm;
    WLEMData::ConstSPtr emd = emm->getModality( m_modality );
    osgAddChannels( *emd );
    osgAddValueGrid( *emd );
    osgSetTrigger( *( emm->getEventChannels() ) );

    WLEMDDrawable2D::osgNodeCallback( nv );
}

void WLEMDDrawable2DSingleChannel::osgSetTrigger( const WLEMMeasurement::EDataT& events )
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
