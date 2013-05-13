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

#include <cstddef>
#include <string>

#include <boost/lexical_cast.hpp>

#include <osg/Geode>
#include <osg/Matrix>
#include <osg/Vec3f>
#include <osg/Vec4f>
#include <osgText/Text>

#include <core/common/WLogger.h>
#include <core/gui/WCustomWidget.h>

#include "core/data/WLDataSetEMM.h"
#include "core/data/emd/WLEMD.h"

#include "WLEMDDrawable2DMultiChannel.h"

namespace LaBP
{
    std::string WLEMDDrawable2DMultiChannel::CLASS = "WLEMDDrawable2DMultiChannel";

    WLEMDDrawable2DMultiChannel::WLEMDDrawable2DMultiChannel( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2D( widget ), m_labelWidth( 32 )
    {
        m_xOffset = m_labelWidth;
        m_channelHeight = 32;
        m_yOffset = m_channelHeight /2;
        m_channelHeightChanged = true;

        m_draw = false;
        // Disable lightning and depth test due to enable opaque colors
        osg::ref_ptr< osg::StateSet > state = m_rootGroup->getOrCreateStateSet();
        state->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
        state->setMode( GL_DEPTH_TEST, osg::StateAttribute::OFF );

        m_labelsBackground = new osg::Geode;
        m_labelsText = new osg::PositionAttitudeTransform;
    }

    WLEMDDrawable2DMultiChannel::~WLEMDDrawable2DMultiChannel()
    {
    }

    WLEMDDrawable::ValueT WLEMDDrawable2DMultiChannel::getChannelHeight() const
    {
        return m_channelHeight;
    }

    void WLEMDDrawable2DMultiChannel::setChannelHeight( ValueT spacing )
    {
        if( m_channelHeight != spacing )
        {
            m_channelHeight = spacing;
            m_yOffset = m_channelHeight /2;
            m_channelHeightChanged = true;
        }
    }

    size_t WLEMDDrawable2DMultiChannel::maxChannels( const LaBP::WLEMD* emd ) const
    {
        size_t channels = ( m_widget->height() / ( m_channelHeight ) );
        channels = channels < emd->getNrChans() ? channels : emd->getNrChans();
        return channels;
    }

    void WLEMDDrawable2DMultiChannel::redraw()
    {
        m_draw = true;
    }

    void WLEMDDrawable2DMultiChannel::osgNodeCallback( osg::NodeVisitor* nv )
    {
        WLEMDDrawable2D::osgNodeCallback( nv );

        m_channelHeightChanged = false;
    }

    void WLEMDDrawable2DMultiChannel::osgAddLabels( const LaBP::WLEMD* emd )
    {
        if( m_labelsText->getNumChildren() != emd->getNrChans() || m_channelHeightChanged )
        {
            // Because GL_DEPTH_TEST is OFF this, order is important:
            // 1. graph data
            // 2. label background
            // 3. labes text
            //
            // NOTE: Solve alpha/transparent/opaque problem to use z coordinate for hiding objects!

            // Create background rectangle for labels //
            osg::ref_ptr< osg::Vec3Array > bgVertices = new osg::Vec3Array;
            const ValueT z = -1.0f;
            bgVertices->push_back( osg::Vec3( 0.0, 0.0, z ) );
            bgVertices->push_back( osg::Vec3( 0.0f, m_widget->height(), z ) );
            bgVertices->push_back( osg::Vec3( m_labelWidth, m_widget->height(), z ) );
            bgVertices->push_back( osg::Vec3( m_labelWidth, 0.0f, z ) );

            osg::ref_ptr< osg::Vec4Array > bgColors = new osg::Vec4Array;
            bgColors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );

            osg::ref_ptr< osg::Geometry > background = new osg::Geometry;
            background->setVertexArray( bgVertices );
            background->setColorArray( bgColors );
            background->setColorBinding( osg::Geometry::BIND_OVERALL );
            background->addPrimitiveSet( new osg::DrawArrays( GL_QUADS, 0, 4 ) );

            m_rootGroup->removeChild( m_labelsBackground );
            m_labelsBackground = new osg::Geode;
            m_labelsBackground->addDrawable( background );
            m_rootGroup->addChild( m_labelsBackground );

            // Create labels //
            m_rootGroup->removeChild( m_labelsText );
            m_labelsText = new osg::PositionAttitudeTransform;
            m_labelsText->setPosition(
                            osg::Vec3d( 0,
                                            m_widget->height() - ( m_channelHeight / 2 ) - ( m_labelWidth / 4 ),
                                            0.0 ) );

            osg::ref_ptr< osg::Geode > labelGeode;
            osg::ref_ptr< osgText::Text > labelText;
            std::string labelName;
            const osg::Vec4 labelColor( 0.0, 0.0, 0.0, 1.0 );
            const size_t channels_begin = 0;
            const size_t channels_count = maxChannels( emd );
            for( size_t channel = channels_begin, channelPos = 0; channelPos < channels_count && channel < emd->getNrChans();
                            ++channel, ++channelPos )
            {
                labelName = boost::lexical_cast< std::string >( channel );
                if( emd->getChanNames().size() > channel )
                {
                    labelName = emd->getChanNames().at( channel );
                }
                labelText = new osgText::Text;
                labelText->setText( labelName );
                labelText->setPosition( osg::Vec3( 0.0, -( float )channelPos * m_channelHeight, 0.0 ) );
                labelText->setAlignment( osgText::Text::LEFT_CENTER );
                labelText->setAxisAlignment( osgText::Text::SCREEN );
                labelText->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
                labelText->setCharacterSize( m_labelWidth / 2 );
                labelText->setColor( labelColor );

                labelGeode = new osg::Geode;
                labelGeode->addDrawable( labelText );
                m_labelsText->addChild( labelGeode );
            }

            m_rootGroup->addChild( m_labelsText );
        }
    }

} /* namespace LaBP */
