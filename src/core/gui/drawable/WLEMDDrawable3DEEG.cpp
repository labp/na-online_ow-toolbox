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

#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/ref_ptr>
#include <osg/ShapeDrawable>
#include <osgText/Text>

#include <core/common/WAssert.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/enum/WLEModality.h"

#include "WLEMDDrawable3D.h"
#include "WLEMDDrawable3DEEG.h"

WLEMDDrawable3DEEG::WLEMDDrawable3DEEG( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable3D( widget )
{
    m_labelsChanged = true;
    m_labelsOn = true;
    m_electrodesChanged = true;
    m_modality = WLEModality::EEG;
}

WLEMDDrawable3DEEG::~WLEMDDrawable3DEEG()
{
    if( m_labesGeode.valid() )
    {
        m_rootGroup->remove( m_labesGeode );
        m_labesGeode = NULL;
    }
    if( m_electrodesGeode.valid() )
    {
        m_rootGroup->remove( m_electrodesGeode );
        m_electrodesGeode = NULL;
    }
}

bool WLEMDDrawable3DEEG::mustDraw() const
{
    return WLEMDDrawable3D::mustDraw() || m_electrodesChanged || m_labelsChanged;
}

void WLEMDDrawable3DEEG::setLabels( bool labelsOn )
{
    if( m_labelsOn != labelsOn )
    {
        m_labelsOn = labelsOn;
        m_labelsChanged = true;
    }
}

void WLEMDDrawable3DEEG::osgAddLabels( const WLPositions& positions, const std::vector< std::string >& labels )
{
    if( m_labelsChanged && m_labelsOn )
    {
        m_rootGroup->removeChild( m_labesGeode );
        m_labesGeode = new osg::Geode;

        const float sphere_size = 4.0f;
        const osg::Vec3 text_offset( 0.0, 0.0, sphere_size );
        const float text_size = 14.0;
        const osg::Vec4 text_color( 0.0, 0.0, 0.0, 1.0 );
        for( WLPositions::IndexT channelID = 0; channelID < positions.size(); ++channelID )
        {
            std::string name = boost::lexical_cast< std::string >( channelID );
            if( labels.size() > channelID )
            {
                name = labels.at( channelID );
            }
            const WLPositions::PositionT tmp = positions.at( channelID ) * 1000; // TODO(pieloth): check factor with unit and exponent.
            osg::Vec3 pos( tmp.x(), tmp.y(), tmp.z() );
            // create text geode for the channel label
            osg::ref_ptr< osgText::Text > text = new osgText::Text;
            text->setText( name );
            text->setPosition( pos + text_offset );
            text->setAlignment( osgText::Text::CENTER_BOTTOM );
            text->setAxisAlignment( osgText::Text::SCREEN );
            text->setCharacterSize( text_size );
            text->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
            text->setColor( text_color );
            m_labesGeode->addDrawable( text );
        }
        m_rootGroup->addChild( m_labesGeode );
    }

    if( m_labelsChanged && !m_labelsOn )
    {
        m_rootGroup->removeChild( m_labesGeode );
    }
    m_labelsChanged = false;
}

void WLEMDDrawable3DEEG::osgAddNodes( const WLPositions& positions )
{
    if( m_electrodesChanged )
    {
        m_electrodesChanged = false;
        m_rootGroup->removeChild( m_electrodesGeode );
        const float sphere_size = 3.0f;
        m_electrodesGeode = new osg::Geode;
        const WLPositions::IndexT count_max = positions.size();
        m_electrodesDrawables.clear();
        m_electrodesDrawables.reserve( count_max );
        for( WLPositions::IndexT channelID = 0; channelID < count_max; ++channelID )
        {
            const WLPositions::PositionT tmp = positions.at( channelID ) * m_zoomFactor; // TODO(pieloth): check factor with unit and exponent.
            osg::Vec3 pos( tmp.x(), tmp.y(), tmp.z() );
            // create sphere geode on electrode position
            osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
            shape->setDataVariance( osg::Object::DYNAMIC );
            m_electrodesDrawables.push_back( shape );
            m_electrodesGeode->addDrawable( shape );
        }
        m_electrodesGeode->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
        m_electrodesGeode->getOrCreateStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );

        m_rootGroup->addChild( m_electrodesGeode );
    }
}

void WLEMDDrawable3DEEG::osgUpdateSurfaceColor( const WLEMData::DataT& data )
{
    if( m_selectedSample >= 0 && ( m_selectedSampleChanged || m_dataChanged || m_colorMapChanged ) )
    {
        if( !m_surfaceGeometry.valid() || m_surfaceGeometry->empty() )
        {
            return;
        }
        osg::ref_ptr< osg::FloatArray > texCoords = static_cast< osg::FloatArray* >( m_surfaceGeometry->getTexCoordArray( 0 ) );

        WAssertDebug( data.rows() == texCoords->size(), "data.rows() == texCoords->size()" );
        WAssertDebug( 0 <= m_selectedSample && m_selectedSample < data.cols(),
                        "0 <= m_selectedSample && m_selectedSample < data.cols()" );
        for( std::size_t vertexID = 0; vertexID < texCoords->size(); ++vertexID )
        {
            float color = data( vertexID, m_selectedSample );
            ( *texCoords )[vertexID] = m_colorMap->getTextureCoordinate( color );
        }
        m_surfaceGeometry->setTexCoordArray( 0, texCoords );
        osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
        colors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );
        m_surfaceGeometry->setColorArray( colors );
        m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
    }
}

void WLEMDDrawable3DEEG::osgUpdateNodesColor( const WLEMData::DataT& data )
{
    if( m_selectedSample >= 0 && ( m_dataChanged || m_colorMapChanged ) )
    {
        float color;
        WAssertDebug( data.rows() == m_electrodesDrawables.size(), "data.rows() == m_electrodesDrawables.size()" );
        WAssertDebug( 0 <= m_selectedSample && m_selectedSample < data.cols(),
                        "0 <= m_selectedSample && m_selectedSample < data.front().size()" );
        for( size_t channelID = 0; channelID < m_electrodesDrawables.size(); ++channelID )
        {
            color = data( channelID, m_selectedSample );
            m_electrodesDrawables.at( channelID )->setColor( m_colorMap->getColor( color ) );
        }
    }
}

void WLEMDDrawable3DEEG::osgNodeCallback( osg::NodeVisitor* nv )
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
    WLEMDEEG::ConstSPtr emd = emm->getModality< const WLEMDEEG >( m_modality );

    if( m_colorMapChanged )
    {
        m_state->setTextureAttributeAndModes( 0, m_colorMap->getAsTexture() );
    }

    const WLPositions& pos = *emd->getChannelPositions3d();
    osgAddLabels( pos, *emd->getChanNames() );

    osgAddNodes( pos );
    osgUpdateNodesColor( emd->getData() );

    osgAddSurface( pos, *emd->getFaces() );
    osgUpdateSurfaceColor( emd->getData() );

    WLEMDDrawable3D::osgNodeCallback( nv );
}
