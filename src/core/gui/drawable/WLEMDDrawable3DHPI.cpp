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

#include <osg/BlendFunc>

#include <core/common/WColor.h> // default color
#include <core/dataHandler/exceptions/WDHException.h>
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>
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
    m_magSensorsGeode = NULL;
    m_colorMap = WLColorMap::SPtr( new WLColorMapClassic( 0, 1, WEColorMapMode::NORMAL ) );
}

WLEMDDrawable3DHPI::~WLEMDDrawable3DHPI()
{
    // TODO Auto-generated destructor stub
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
    if( m_magSensorsGeode.valid() )
    {
        return;
    }

    if( !m_emm->hasModality( WLEModality::MEG ) )
    {
        wlog::error( CLASS ) << "No MEG data!";
        return;
    }

    WLEMDMEG::ConstSPtr megIn = m_emm->getModality< const WLEMDMEG >( WLEModality::MEG );
    WLEMDMEG::SPtr magOut;
    if( !WLEMDMEG::extractCoilModality( magOut, megIn, WLEModality::MEG_MAG, false ) )
    {
        wlog::error( CLASS ) << "Could not extract magnetometer!";
        return;
    }
    const std::vector< WPosition >& positions = *( magOut->getChannelPositions3d() );
    const std::vector< WVector3i >& faces = *( magOut->getFaces() );

    // osgAddMegNodes
    // --------------
    const WColor mag_node = defaultColor::BLUE;
    m_rootGroup->removeChild( m_magSensorsGeode );
    const float sphere_size = 1.0f;
    m_magSensorsGeode = new osg::Geode;
    m_magSensorsDrawables.clear();
    m_magSensorsDrawables.reserve( positions.size() );
    std::vector< WPosition >::const_iterator it;
    for( it = positions.begin(); it != positions.end(); ++it )
    {
        const osg::Vec3 pos = *it * m_zoomFactor;
        // create sphere geode on electrode position
        osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
        shape->setDataVariance( osg::Object::DYNAMIC );
        shape->setColor( mag_node );
        m_magSensorsDrawables.push_back( shape );
        m_magSensorsGeode->addDrawable( shape );
    }
    m_magSensorsGeode->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_magSensorsGeode->getOrCreateStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );

    m_rootGroup->addChild( m_magSensorsGeode );

    // osgAddSurface
    // -------------
    const WColor mag_surface( 0.9, 0.9, 9.0, 0.5 );
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
    // TODO(pieloth): Optimize transparency
    m_surfaceGeometry = wge::convertToOsgGeometry( tri, mag_surface, false, false, false );
    m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
    m_state->setMode( GL_BLEND, osg::StateAttribute::ON );
    m_state->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_state->setMode( GL_NORMALIZE, osg::StateAttribute::ON );
    m_state->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    m_surfaceGeometry->setStateSet( m_state );
    m_surfaceGeometry->setDataVariance( osg::Object::STATIC );

    m_surfaceGeode = new osg::Geode;
    m_surfaceGeode->addDrawable( m_surfaceGeometry );
    m_rootGroup->addChild( m_surfaceGeode );
}

void WLEMDDrawable3DHPI::osgAddOrUpdateHpiCoils( const std::vector< WPosition >& positions )
{
    m_rootGroup->removeChild( m_hpiCoilsGeode );
    const float sphere_size = 3.0f;
    m_hpiCoilsGeode = new osg::Geode;
    m_hpiCoilsDrawables.clear();
    m_hpiCoilsDrawables.reserve( positions.size() );
    std::vector< WPosition >::const_iterator it;
    for( it = positions.begin(); it != positions.end(); ++it )
    {
        const osg::Vec3 pos = *it * m_zoomFactor;
        // create sphere geode on electrode position
        osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
        shape->setColor( defaultColor::RED );
        shape->setDataVariance( osg::Object::DYNAMIC );
        m_hpiCoilsDrawables.push_back( shape );
        m_hpiCoilsGeode->addDrawable( shape );
    }
    m_hpiCoilsGeode->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
    m_hpiCoilsGeode->getOrCreateStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );

    m_rootGroup->addChild( m_hpiCoilsGeode );
}
