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

#include <vector>

#include <osg/BoundingBox>
#include <osg/LightModel>
#include <osg/ShapeDrawable>

#include <core/common/WAssert.h>
#include <core/common/WColor.h> // default color
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>
#include <core/graphicsEngine/WGENoOpManipulator.h>
#include <core/graphicsEngine/WTriangleMesh.h>

#include "core/data/emd/WLEMDHPI.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/gui/colorMap/WLColorMapClassic.h"
#include "core/util/WLGeometry.h"
#include "WLEMDDrawable3DHPI.h"

using namespace LaBP;

WLEMDDrawable3DHPI::WLEMDDrawable3DHPI( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable3D( widget )
{
    m_modality = WLEModality::HPI;
    m_viewTransformation = osg::Matrixd::identity();
    m_magSensorsGeode = NULL;
    m_colorMap = WLColorMap::SPtr( new WLColorMapClassic( 0, 1, WEColorMapMode::NORMAL ) );
    m_zoomFactor = 800.0;
    widget->getViewer()->setCameraManipulator( new WGENoOpManipulator );
}

WLEMDDrawable3DHPI::~WLEMDDrawable3DHPI()
{
}

void WLEMDDrawable3DHPI::setView( View view )
{
    m_viewTransformation = getTransformation( view );
}

void WLEMDDrawable3DHPI::osgNodeCallback( osg::NodeVisitor* nv )
{
    if( !mustDraw() )
    {
        return;
    }

    if( !hasData() )
    {
        return;
    }

    osgInitMegHelmet();

    WLEMDHPI::ConstSPtr emdHpi = m_emm->getModality< const WLEMDHPI >( WLEModality::HPI );
    WLArrayList< WPosition >::ConstSPtr positions = emdHpi->getChannelPositions3d();
    WLArrayList< WLEMDHPI::TransformationT >::ConstSPtr transformations = emdHpi->getTransformations();
    std::vector< WPosition > posNew;
    posNew.reserve( positions->size() );
    WLGeometry::transformPoints( &posNew, *positions, transformations->at( 0 ) );
    osgAddOrUpdateHpiCoils( posNew );

    WLEMDDrawable3D::osgNodeCallback( nv );
    m_rootGroup->removeChild( m_colorMapNode );
}

void WLEMDDrawable3DHPI::osgInitMegHelmet()
{
    if( m_viewGeode.valid() )
    {
        return;
    }

    if( !m_emm->hasModality( WLEModality::MEG ) )
    {
        wlog::error( CLASS ) << "No MEG data!";
        return;
    }

    // Prepare data
    WLEMDMEG::ConstSPtr megIn = m_emm->getModality< const WLEMDMEG >( WLEModality::MEG );
    WLEMDMEG::SPtr magOut;
    if( !WLEMDMEG::extractCoilModality( magOut, megIn, WLEModality::MEG_MAG, false ) )
    {
        wlog::error( CLASS ) << "Could not extract magnetometer!";
        return;
    }
    const std::vector< WPosition >& positions = *( magOut->getChannelPositions3d() );
    const std::vector< WVector3i >& faces = *( magOut->getFaces() );

    // Prepare view: top, side, buttom
    m_rootGroup->removeChild( m_viewGeode );
    m_viewGeode = new osg::MatrixTransform;
    m_viewGeode->setMatrix( m_viewTransformation );

    // osgAddMegNodes
    // --------------
    const WColor mag_node = defaultColor::BLUE;
    const float sphere_size = 1.0f;
    m_magSensorsGeode = new osg::Geode;
    std::vector< WPosition >::const_iterator it;
    for( it = positions.begin(); it != positions.end(); ++it )
    {
        const osg::Vec3 pos = *it * m_zoomFactor;
        // create sphere geode on electrode position
        osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
        shape->setColor( mag_node );
        m_magSensorsGeode->addDrawable( shape );
    }
    m_magSensorsGeode->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_magSensorsGeode->getOrCreateStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );

    m_viewGeode->addChild( m_magSensorsGeode );

    // osgAddSurface
    // -------------
    const WColor mag_surface( 0.9, 0.9, 9.0, 0.6 );
    const size_t nbPositions = positions.size();
    std::vector< WPosition > scaledPos;
    scaledPos.reserve( nbPositions );
    for( size_t i = 0; i < nbPositions; ++i )
    {
        scaledPos.push_back( positions[i] * m_zoomFactor );
    }
    WTriangleMesh::SPtr tri;
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

        tri = WTriangleMesh::SPtr( new WTriangleMesh( vertices, triangles ) );
    }
    else
    {
        try
        {
            tri = wge::triangulate( scaledPos, -0.005 );
        }
        catch( WException& e )
        {
            wlog::error( CLASS ) << "wge::triangulate() " << e.what();
            return;
        }
    }

    m_surfaceGeometry = wge::convertToOsgGeometry( tri, mag_surface, true, false, false );
    m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );

    m_surfaceGeode = new osg::Geode;
    wge::enableTransparency( m_surfaceGeode );
    osg::ref_ptr< osg::LightModel > lightModel = new osg::LightModel;
    lightModel->setTwoSided( true );
    m_surfaceGeode->getStateSet()->setAttributeAndModes( lightModel );
    m_surfaceGeode->getStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );
    m_surfaceGeode->getStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_surfaceGeode->addDrawable( m_surfaceGeometry );

    m_viewGeode->addChild( m_surfaceGeode );
    m_rootGroup->addChild( m_viewGeode );
}

void WLEMDDrawable3DHPI::osgAddOrUpdateHpiCoils( const std::vector< WPosition >& positions )
{
    m_viewGeode->removeChild( m_hpiCoilsGeode );
    const float sphere_size = 3.0f;
    m_hpiCoilsGeode = new osg::Geode;
    std::vector< WPosition >::const_iterator it;
    for( it = positions.begin(); it != positions.end(); ++it )
    {
        const osg::Vec3 pos = *it * m_zoomFactor;
        // create sphere geode on hpi position
        osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
        shape->setColor( defaultColor::DARKRED );
        shape->setInitialBound( osg::BoundingBox( 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ) );
        m_hpiCoilsGeode->addDrawable( shape );
    }

    m_viewGeode->addChild( m_hpiCoilsGeode );
}

osg::Matrixd WLEMDDrawable3DHPI::getTransformation( View view )
{
    switch( view )
    {
        case VIEW_TOP:
            return osg::Matrixd::identity();
        case VIEW_SIDE:
            return osg::Matrixd::rotate( M_PI / -2, osg::Vec3d( 1, 0, 0 ) )
                            * osg::Matrixd::rotate( M_PI / 2, osg::Vec3d( 0, 1, 0 ) );
        case VIEW_FRONT:
            return osg::Matrixd::rotate( M_PI / -2, osg::Vec3d( 1, 0, 0 ) ) * osg::Matrixd::rotate( M_PI, osg::Vec3d( 0, 1, 0 ) );
        default:
            WAssert( false, "View not supported!" );
            return osg::Matrixd::identity();
    }
}
