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

#include <sstream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/Geode>
#include <osg/LightModel>
#include <osg/ref_ptr>
#include <osg/Drawable>
#include <osg/Texture>
#include <osgText/Text>

#include <core/common/WException.h>
#include <core/common/WLogger.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/dataHandler/exceptions/WDHException.h>
#include <core/graphicsEngine/WTriangleMesh.h>
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>
#include <core/graphicsEngine/WGEZoomTrackballManipulator.h>

#include "core/data/emd/WLEMDMEG.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/util/WLDefines.h"

#include "WLEMDDrawable.h"
#include "WLEMDDrawable3DEEG.h"
#include "WLEMDDrawable3DMEG.h"
#include "WLEMDDrawable3DSource.h"
#include "WLEMDDrawable3DEmpty.h"
#include "WLEMDDrawable3D.h"

const std::string WLEMDDrawable3D::CLASS = "WLEMDDrawable3D";

WLEMDDrawable3D::WLEMDDrawable3D( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable( widget )
{
    widget->getViewer()->setCameraManipulator( new WGEZoomTrackballManipulator );

    m_zoomFactor = 1000;
    m_selectedSample = -1;
    m_selectedSampleChanged = false;
    m_colorMapChanged = true;
    m_surfaceChanged = true;
    m_state = new osg::StateSet;
    osg::ref_ptr< osg::LightModel > lightModel = new osg::LightModel;
    lightModel->setTwoSided( true );
    m_state->setAttributeAndModes( lightModel );
    m_state->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_state->setMode( GL_NORMALIZE, osg::StateAttribute::ON );
}

WLEMDDrawable3D::~WLEMDDrawable3D()
{
}

WLEMDDrawable3D::SPtr WLEMDDrawable3D::getInstance( WUIViewWidget::SPtr widget, WLEModality::Enum modality )
{
    WLEMDDrawable3D::SPtr drawable3D;
    switch( modality )
    {
        case WLEModality::EEG:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DEEG( widget ) );
            break;
        case WLEModality::MEG:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DMEG( widget ) );
            break;
        case WLEModality::MEG_MAG:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DMEG( widget, modality ) );
            break;
        case WLEModality::MEG_GRAD:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DMEG( widget, modality ) );
            break;
        case WLEModality::MEG_GRAD_MERGED:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DMEG( widget, modality ) );
            break;
        case WLEModality::SOURCE:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DSource( widget ) );
            break;
        default:
            drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DEmpty( widget ) );
            wlog::warn( CLASS ) << "No 3D drawable available for modality: " << WLEModality::name( modality );
            break;
    }
    if( WLEModality::isMEGCoil( modality ) )
    {
        modality = WLEModality::MEG;
    }
    drawable3D->setModality( modality );
    return drawable3D;
}

bool WLEMDDrawable3D::mustDraw() const
{
    return WLEMDDrawable::mustDraw() || m_colorMapChanged || m_selectedSampleChanged || m_surfaceChanged;
}

void WLEMDDrawable3D::draw( WLEMMeasurement::SPtr emm )
{
    m_emm = emm;
    m_dataChanged = true;
    redraw();
}

void WLEMDDrawable3D::osgNodeCallback( osg::NodeVisitor* nv )
{
    WL_UNUSED( nv );
    osgAddColorMap();

    m_colorMapChanged = false;
    m_selectedSampleChanged = false;
    m_surfaceChanged = false;

    WLEMDDrawable::resetDrawFlags();
}

bool WLEMDDrawable3D::hasData() const
{
    return m_emm.get() && m_emm->hasModality( m_modality );
}

ptrdiff_t WLEMDDrawable3D::getSelectedSample() const
{
    return m_selectedSample;
}

bool WLEMDDrawable3D::setSelectedSample( ptrdiff_t sample )
{
    if( sample != m_selectedSample )
    {
        m_selectedSample = sample;
        m_selectedSampleChanged = true;
        return true;
    }
    return false;
}

float WLEMDDrawable3D::getSelectedTime() const
{
    if( m_emm && m_emm->hasModality( m_modality ) )
    {
        const size_t size = m_emm->getModality( m_modality )->getSamplesPerChan();
        return ( float )m_selectedSample / ( float )size;
    }
    else
    {
        return -1.0;
    }
}

bool WLEMDDrawable3D::setSelectedTime( float relative )
{
    if( relative < 0 )
    {
        return false;
    }
    if( m_emm && m_emm->hasModality( m_modality ) )
    {
        const float pos = relative * m_emm->getModality( m_modality )->getSamplesPerChan();
        wlog::debug( CLASS ) << "setSelectedTime(): " << pos;
        return setSelectedSample( pos );
    }
    else
    {
        return false;
    }
}

WLColorMap::SPtr WLEMDDrawable3D::getColorMap() const
{
    return m_colorMap;
}

void WLEMDDrawable3D::setColorMap( WLColorMap::SPtr colorMap )
{
    m_colorMap = colorMap;
    m_colorMapChanged = true;
}

void WLEMDDrawable3D::osgAddSurface( const std::vector< WPosition >& positions, const std::vector< WVector3i >& faces )
{
    // draw head surface
    if( m_surfaceChanged )
    {
        m_rootGroup->removeChild( m_surfaceGeode );

        const size_t nbPositions = positions.size();
        std::vector< WPosition > scaledPos;
        scaledPos.reserve( nbPositions );
        for( size_t i = 0; i < nbPositions; ++i )
        {
            scaledPos.push_back( positions[i] * m_zoomFactor );
        }
        boost::shared_ptr< WTriangleMesh > tri;
        if( faces.size() > 0 )
        {
            osg::ref_ptr< osg::Vec3Array > vertices = wge::osgVec3Array( scaledPos );
            std::vector< size_t > triangles;
            triangles.resize( faces.size() * 3 );

            for( size_t i = 0; i < faces.size(); ++i )
            {
                triangles.push_back( faces.at( i ).x() );
                triangles.push_back( faces.at( i ).y() );
                triangles.push_back( faces.at( i ).z() );
            }

            tri = boost::shared_ptr< WTriangleMesh >( new WTriangleMesh( vertices, triangles ) );
            m_surfaceGeometry = wge::convertToOsgGeometry( tri, WColor( 1.0, 1.0, 1.0, 1.0 ), true );
        }
        else
        {
            try
            {
                tri = wge::triangulate( scaledPos, -0.005 );
            }
            catch( const WException& e )
            {
                wlog::error( CLASS ) << "wge::triangulate() " << e.what();
                return;
            }
            m_surfaceGeometry = wge::convertToOsgGeometry( tri, WColor( 1.0, 1.0, 1.0, 1.0 ), true );
        }
        osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
        colors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );
        m_surfaceGeometry->setColorArray( colors );
        m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
        m_surfaceGeometry->setStateSet( m_state );
        osg::ref_ptr< osg::FloatArray > texCoords = new osg::FloatArray;
        texCoords->assign( nbPositions, 0.5f );
        m_surfaceGeometry->setTexCoordArray( 0, texCoords );
        m_surfaceGeometry->setDataVariance( osg::Object::DYNAMIC );

        m_surfaceGeode = new osg::Geode;
        m_surfaceGeode->addDrawable( m_surfaceGeometry );
        m_rootGroup->addChild( m_surfaceGeode );
    }
}

void WLEMDDrawable3D::osgAddColorMap()
{
    if( m_colorMapChanged )
    {
        m_rootGroup->removeChild( m_colorMapNode );

        const ValueT width = m_widget->width();
        const ValueT height = m_widget->height();
        const WLColorMap::ValueT cm_max = m_colorMap->getMax();
        const WLColorMap::ValueT cm_min = m_colorMap->getMin();
        const ValueT cm_width = 20;
        const ValueT cm_height = height / 2;
        const ValueT cm_x_offset = 10;
        const ValueT cm_y_offset = cm_height / 2;

        osg::ref_ptr< osg::MatrixTransform > cmModelViewMat;
        osg::ref_ptr< osg::Geode > cmGeode;

        // Prepare projection and matrix for fixed position and scale
        {
            m_colorMapNode = new osg::Projection;
            m_colorMapNode->setMatrix( osg::Matrix::ortho2D( 0, width, 0, height ) );

            // For the HUD model view matrix use an identity matrix:
            cmModelViewMat = new osg::MatrixTransform;
            cmModelViewMat->setMatrix( osg::Matrix::identity() );

            // Make sure the model view matrix is not affected by any transforms
            // above it in the scene graph:
            cmModelViewMat->setReferenceFrame( osg::Transform::ABSOLUTE_RF );

            cmModelViewMat->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
        }
        // Add the HUD projection matrix as a child of the root node
        // and the HUD model view matrix as a child of the projection matrix
        // Anything under this node will be viewed using this projection matrix
        // and positioned with this model view matrix.
        m_colorMapNode->addChild( cmModelViewMat );

        cmGeode = new osg::Geode;
        cmModelViewMat->addChild( cmGeode );

        // Draw Color Map Bar
        {
            osg::ref_ptr< osg::Geometry > quadGeom = new osg::Geometry;

            // Coordinates Counter Clock Wise (CCW)
            osg::ref_ptr< osg::Vec3Array > quadVerts = new osg::Vec3Array();
            quadVerts->reserve( 4 );
            quadVerts->push_back( osg::Vec3( cm_x_offset, cm_y_offset + cm_height, 0.0 ) );
            quadVerts->push_back( osg::Vec3( cm_x_offset, cm_y_offset, 0.0 ) );
            quadVerts->push_back( osg::Vec3( cm_x_offset + cm_width, cm_y_offset, 0.0 ) );
            quadVerts->push_back( osg::Vec3( cm_x_offset + cm_width, cm_y_offset + cm_height, 0.0 ) );
            quadGeom->setVertexArray( quadVerts );

            osg::ref_ptr< osg::FloatArray > quadTexCoords = new osg::FloatArray;
            quadTexCoords->assign( 4, 0.0 );
            ( *quadTexCoords )[0] = m_colorMap->getTextureCoordinate( cm_max );
            ( *quadTexCoords )[1] = m_colorMap->getTextureCoordinate( cm_min );
            ( *quadTexCoords )[2] = m_colorMap->getTextureCoordinate( cm_min );
            ( *quadTexCoords )[3] = m_colorMap->getTextureCoordinate( cm_max );
            quadGeom->setTexCoordArray( 0, quadTexCoords );

            quadGeom->getOrCreateStateSet()->setTextureAttributeAndModes( 0, m_colorMap->getAsTexture() );
            quadGeom->setColorBinding( osg::Geometry::BIND_PER_VERTEX );
            quadGeom->addPrimitiveSet( new osg::DrawArrays( GL_QUADS, 0, quadVerts->size() ) );

            cmGeode->addDrawable( quadGeom );
        }

        // Add captions to Color Map Bar
        {
            osg::ref_ptr< osgText::Text > text;
            std::stringstream cmValueStream;
            const ValueT cmt_x = cm_x_offset + cm_width + 5;
            const osg::Vec4 cmt_color( 0.0, 0.0, 0.0, 1.0 );
            const float cmt_char_size = 16;

            cmValueStream.clear();
            cmValueStream.str( "" );
            cmValueStream << std::setprecision( 2 ) << std::scientific << cm_max;
            text = new osgText::Text;
            text->setText( cmValueStream.str() );
            text->setPosition( osg::Vec3( cmt_x, cm_y_offset + cm_height, 0.0 ) );
            text->setAlignment( osgText::Text::LEFT_CENTER );
            text->setAxisAlignment( osgText::Text::SCREEN );
            text->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
            text->setCharacterSize( cmt_char_size );
            text->setColor( cmt_color );
            cmGeode->addDrawable( text );

            cmValueStream.clear();
            cmValueStream.str( "" );
            cmValueStream << std::setprecision( 2 ) << cm_min + ( cm_max - cm_min ) / 2;
            text = new osgText::Text;
            text->setText( cmValueStream.str() );
            text->setPosition( osg::Vec3( cmt_x, cm_y_offset + cm_height / 2, 0.0 ) );
            text->setAlignment( osgText::Text::LEFT_CENTER );
            text->setAxisAlignment( osgText::Text::SCREEN );
            text->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
            text->setCharacterSize( cmt_char_size );
            text->setColor( cmt_color );
            cmGeode->addDrawable( text );

            cmValueStream.clear();
            cmValueStream.str( "" );
            cmValueStream << std::setprecision( 2 ) << std::scientific << cm_min;
            text = new osgText::Text;
            text->setText( cmValueStream.str() );
            text->setPosition( osg::Vec3( cmt_x, cm_y_offset, 0.0 ) );
            text->setAlignment( osgText::Text::LEFT_CENTER );
            text->setAxisAlignment( osgText::Text::SCREEN );
            text->setCharacterSizeMode( osgText::Text::SCREEN_COORDS );
            text->setCharacterSize( cmt_char_size );
            text->setColor( cmt_color );
            cmGeode->addDrawable( text );
        }

        m_rootGroup->addChild( m_colorMapNode );
    }
}
