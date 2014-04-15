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

#include <boost/lexical_cast.hpp>

#include <osg/Array>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osgText/Text>

#include <core/common/WAssert.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMDMEG.h"

#include "WLEMDDrawable3D.h"
#include "WLEMDDrawable3DMEG.h"

namespace LaBP
{
    WLEMDDrawable3DMEG::WLEMDDrawable3DMEG( WUIViewWidget::SPtr widget, WLEModality::Enum coilType ) :
                    m_coilType( coilType ), WLEMDDrawable3D( widget )
    {
        WAssertDebug( WLEModality::isMEG(m_coilType), "No MEG modality!" );
        m_labelsChanged = true;
        m_labelsOn = true;
        m_electrodesChanged = true;
        m_modality = WLEModality::MEG;
    }

    WLEMDDrawable3DMEG::WLEMDDrawable3DMEG( WUIViewWidget::SPtr widget ) :
                    m_coilType( WLEModality::MEG ), WLEMDDrawable3D( widget )
    {
        m_labelsChanged = true;
        m_labelsOn = true;
        m_electrodesChanged = true;
        m_modality = WLEModality::MEG;
    }

    WLEMDDrawable3DMEG::~WLEMDDrawable3DMEG()
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

    bool WLEMDDrawable3DMEG::mustDraw() const
    {
        return WLEMDDrawable3D::mustDraw() || m_electrodesChanged || m_labelsChanged;
    }

    void WLEMDDrawable3DMEG::setLabels( bool labelsOn )
    {
        if( m_labelsOn != labelsOn )
        {
            m_labelsOn = labelsOn;
            m_labelsChanged = true;
        }
    }

    void WLEMDDrawable3DMEG::osgAddLabels( const std::vector< WPosition >& positions, const std::vector< std::string >& labels )
    {
        if( m_labelsChanged && m_labelsOn )
        {
            m_rootGroup->removeChild( m_labesGeode );
            m_labesGeode = new osg::Geode;

            const float sphere_size = 4.0f;
            const osg::Vec3 text_offset( 0.0, 0.0, sphere_size );
            const float text_size = 14.0;
            const osg::Vec4 text_color( 0.0, 0.0, 0.0, 1.0 );

            for( size_t i = 0; i < positions.size(); ++i )
            {
                std::string name;
                if( i < labels.size() )
                {
                    name = labels.at( i );
                }
                else
                {
                    name = boost::lexical_cast< std::string >( i );
                }
                osg::Vec3 pos = positions.at( i ) * 1000;
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

    void WLEMDDrawable3DMEG::osgAddNodes( const std::vector< WPosition >& positions )
    {
        if( m_electrodesChanged )
        {
            m_electrodesChanged = false;
            m_rootGroup->removeChild( m_electrodesGeode );
            const float sphere_size = 3.0f;
            m_electrodesGeode = new osg::Geode;
            m_electrodesDrawables.clear();
            m_electrodesDrawables.reserve( positions.size() );
            std::vector< WPosition >::const_iterator it;
            for( it = positions.begin(); it != positions.end(); ++it )
            {
                const osg::Vec3 pos = *it * m_zoomFactor;
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

    void WLEMDDrawable3DMEG::osgUpdateSurfaceColor( const WLEMData::DataT& data )
    {
        if( m_selectedSample >= 0 && ( m_selectedSampleChanged || m_dataChanged || m_colorMapChanged ) )
        {
            if( !m_surfaceGeometry.valid() || m_surfaceGeometry->empty() )
            {
                return;
            }
            osg::ref_ptr< osg::FloatArray > texCoords =
                            static_cast< osg::FloatArray* >( m_surfaceGeometry->getTexCoordArray( 0 ) );

            WAssertDebug( data.rows() == texCoords->size(), "data.rows() == texCoords->size()" );
            WAssertDebug( 0 <= m_selectedSample && m_selectedSample < data.cols(),
                            "0 <= m_selectedSample && m_selectedSample < data.cols()" );
            for( std::size_t vertexID = 0; vertexID < texCoords->size(); ++vertexID )
            {
                const float color = data( vertexID, m_selectedSample );
                ( *texCoords )[vertexID] = m_colorMap->getTextureCoordinate( color );
            }
            m_surfaceGeometry->setTexCoordArray( 0, texCoords );
            osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
            colors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );
            m_surfaceGeometry->setColorArray( colors );
            m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
        }
    }

    void WLEMDDrawable3DMEG::osgUpdateNodesColor( const WLEMData::DataT& data )
    {
        if( m_selectedSample >= 0 && ( m_dataChanged || m_colorMapChanged ) )
        {
            WAssertDebug( data.rows() == m_electrodesDrawables.size(), "data.rows() == m_electrodesDrawables.size()" );
            WAssertDebug( 0 <= m_selectedSample && m_selectedSample < data.cols(),
                            "0 <= m_selectedSample && m_selectedSample < data.front().size()" );
            for( size_t channelID = 0; channelID < m_electrodesDrawables.size(); ++channelID )
            {
                const float color = data( channelID, m_selectedSample );
                m_electrodesDrawables.at( channelID )->setColor( m_colorMap->getColor( color ) );
            }
        }
    }

    void WLEMDDrawable3DMEG::osgNodeCallback( osg::NodeVisitor* nv )
    {
        if( !mustDraw() )
        {
            return;
        }

        if( !hasData() )
        {
            return;
        }

        WLEMDMEG::ConstSPtr emd = m_emm->getModality< const WLEMDMEG >( WLEModality::MEG );

        if( WLEModality::isMEGCoil( m_coilType ) )
        {
            WLEMDMEG::SPtr meg;
            if( WLEMDMEG::extractCoilModality( meg, emd, m_coilType, false ) )
            {
                emd = meg;
            }
            else
            {
                return;
            }
        }

        if( m_colorMapChanged )
        {
            m_state->setTextureAttributeAndModes( 0, m_colorMap->getAsTexture() );
        }

        osgAddLabels( *emd->getChannelPositions3d(), *emd->getChanNames() );

        osgAddNodes( *emd->getChannelPositions3d() );
        osgUpdateNodesColor( emd->getData() );

        if( m_surfaceChanged )
        {
            osgAddSurface( *emd->getChannelPositions3d(), *emd->getFaces() );
        }
        osgUpdateSurfaceColor( emd->getData() );

        WLEMDDrawable3D::osgNodeCallback( nv );
    }
} /* namespace LaBP */
