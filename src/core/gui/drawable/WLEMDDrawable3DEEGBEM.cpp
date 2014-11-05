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

#include <list>
#include <vector>

#include <core/common/WColor.h>
#include <core/common/WLogger.h>
#include <core/graphicsEngine/WTriangleMesh.h>
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/gui/colorMap/WLColorMap.h"
#include "core/gui/colorMap/WLColorMapClassic.h"
#include "core/util/WLGeometry.h"

#include "WLEMDDrawable3DEEGBEM.h"

WLEMDDrawable3DEEGBEM::WLEMDDrawable3DEEGBEM( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable3D( widget )
{
    m_electrodesChanged = true;
    m_modality = WLEModality::EEG;
    m_colorMap = WLColorMap::SPtr( new WLColorMapClassic( 0, 1, WEColorMapMode::NORMAL ) );
}

WLEMDDrawable3DEEGBEM::~WLEMDDrawable3DEEGBEM()
{
    if( m_electrodesGeode.valid() )
    {
        m_rootGroup->remove( m_electrodesGeode );
        m_electrodesGeode = NULL;
    }
}

bool WLEMDDrawable3DEEGBEM::mustDraw() const
{
    return WLEMDDrawable3D::mustDraw() || m_electrodesChanged;
}

void WLEMDDrawable3DEEGBEM::osgNodeCallback( osg::NodeVisitor* nv )
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
    WLEMDEEG::ConstSPtr emd = emm->getModality< const WLEMDEEG >( WLEModality::EEG );

    WLEMMSubject::ConstSPtr subject = emm->getSubject();
    const std::list< WLEMMBemBoundary::SPtr >& bems = *subject->getBemBoundaries();
    std::list< WLEMMBemBoundary::SPtr >::const_iterator itBem;
    WLEMMBemBoundary::ConstSPtr bemSkin;
    for( itBem = bems.begin(); itBem != bems.end(); ++itBem )
    {
        if( ( *itBem )->getBemType() == WLEBemType::OUTER_SKIN || ( *itBem )->getBemType() == WLEBemType::HEAD )
        {
            bemSkin = *itBem;
            break;
        }
    }

    std::vector< WPosition > bemPositions;
    WLGeometry::toBaseExponent( &bemPositions, *bemSkin->getVertex(), bemSkin->getVertexExponent() );
    const std::vector< WVector3i >& bemFaces = *bemSkin->getFaces();

    osgAddSurface( bemPositions, bemFaces );

    std::vector< WPosition > pointsTrans;
    WLGeometry::transformPoints( &pointsTrans, *( emd->getChannelPositions3d() ), emm->getFidToACPCTransformation() );

    osgAddNodes( pointsTrans );

    WLEMDDrawable3D::osgNodeCallback( nv );
    m_rootGroup->removeChild( m_colorMapNode );
}

void WLEMDDrawable3DEEGBEM::osgAddNodes( const std::vector< WPosition >& positions )
{
    if( m_electrodesChanged )
    {
        m_electrodesChanged = false;
        m_rootGroup->removeChild( m_electrodesGeode );
        const float sphere_size = 3.0f;
        m_electrodesGeode = new osg::Geode;

        const size_t count_max = positions.size();
        m_electrodesDrawables.clear();
        m_electrodesDrawables.reserve( count_max );
        for( size_t channelID = 0; channelID < count_max; ++channelID )
        {
            osg::Vec3 pos = positions.at( channelID ) * m_zoomFactor;
            // create sphere geode on electrode position
            osg::ref_ptr< osg::ShapeDrawable > shape = new osg::ShapeDrawable( new osg::Sphere( pos, sphere_size ) );
            shape->setDataVariance( osg::Object::DYNAMIC );
            shape->setColor( defaultColor::DARKRED );
            m_electrodesDrawables.push_back( shape );
            m_electrodesGeode->addDrawable( shape );
        }
        m_electrodesGeode->getOrCreateStateSet()->setMode( GL_LIGHTING, osg::StateAttribute::ON );
        m_electrodesGeode->getOrCreateStateSet()->setMode( GL_NORMALIZE, osg::StateAttribute::ON );

        m_rootGroup->addChild( m_electrodesGeode );
    }
}

void WLEMDDrawable3DEEGBEM::osgAddSurface( const std::vector< WPosition >& positions, const std::vector< WVector3i >& faces )
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
            m_surfaceGeometry = wge::convertToOsgGeometry( tri, defaultColor::LIGHTYELLOW, true );
        }
        else
        {
            tri = wge::triangulate( scaledPos, -0.005 );
            m_surfaceGeometry = wge::convertToOsgGeometry( tri, defaultColor::LIGHTYELLOW, true );
        }
        osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
        colors->push_back( osg::Vec4( defaultColor::LIGHTYELLOW ) );
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
