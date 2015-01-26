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

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <osg/Array>
#include <osg/ref_ptr>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/WLogger.h>
#include <core/graphicsEngine/WGEGeodeUtils.h> // osgAddSurface
#include <core/graphicsEngine/WTriangleMesh.h> // osgAddSurface
#include <core/kernel/WROIManager.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/gui/roi/WLROIBox.h"

#include "WLEMDDrawable3DSource.h"

WLEMDDrawable3DSource::WLEMDDrawable3DSource( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable3D( widget )
{
    m_zoomFactor = 1;

    // Properties
    m_properties = boost::shared_ptr< WProperties >( new WProperties( "Properties", "View's properties" ) );
    m_trgNewRoi = m_properties->addProperty( "New ROI", "Insert a new ROI.", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WLEMDDrawable3DSource::callbackNewRoi_Clicked, this ) );
    m_widget->addAction( m_trgNewRoi );

    //drawCoords();
}

WLEMDDrawable3DSource::~WLEMDDrawable3DSource()
{
}

void WLEMDDrawable3DSource::osgUpdateSurfaceColor( const WLEMData::DataT& data )
{
    if( m_selectedSample >= 0 && ( m_selectedSampleChanged || m_dataChanged || m_colorMapChanged ) )
    {
        float color;
        osg::ref_ptr< osg::FloatArray > texCoords = static_cast< osg::FloatArray* >( m_surfaceGeometry->getTexCoordArray( 0 ) );
        WAssertDebug( data.rows() == texCoords->size(), "data.rows() == texCoords->size()" );
        WAssertDebug( 0 <= m_selectedSample && m_selectedSample < data.cols(),
                        "0 <= m_selectedSample && m_selectedSample < data.cols()" );
        for( std::size_t vertexID = 0; vertexID < texCoords->size(); ++vertexID )
        {
            color = data( vertexID, m_selectedSample );
            ( *texCoords )[vertexID] = m_colorMap->getTextureCoordinate( color );
        }
        m_surfaceGeometry->setTexCoordArray( 0, texCoords );
        osg::ref_ptr< osg::Vec4Array > colors = new osg::Vec4Array;
        colors->push_back( osg::Vec4( 1.0f, 1.0f, 1.0f, 1.0f ) );
        m_surfaceGeometry->setColorArray( colors );
        m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_OVERALL );
    }
}

void WLEMDDrawable3DSource::osgNodeCallback( osg::NodeVisitor* nv )
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
    WLEMDSource::ConstSPtr emd = emm->getModality< const WLEMDSource >( WLEModality::SOURCE );
    WLEMMSubject::ConstSPtr subject = emm->getSubject();

    if( m_colorMapChanged )
    {
        m_state->setTextureAttributeAndModes( 0, m_colorMap->getAsTexture() );
    }

    WLEMMSurface::ConstSPtr surf = subject->getSurface( WLEMMSurface::Hemisphere::BOTH );

    m_zoomFactor = WLEExponent::factor( surf->getVertexExponent() ) * 1000;
    osgAddSurface( *surf->getVertex(), *surf->getFaces() );

    osgUpdateSurfaceColor( emd->getData() );

    WLEMDDrawable3D::osgNodeCallback( nv );
}

void WLEMDDrawable3DSource::setROISelector( WLEMDDrawable3DSource::ROISelectorSPtr roiSelector )
{
    m_roiSelecor = roiSelector;
}

WLEMDDrawable3DSource::ROISelectorSPtr WLEMDDrawable3DSource::getROISelector()
{
    return m_roiSelecor;
}

void WLEMDDrawable3DSource::callbackNewRoi_Clicked()
{
    // TODO(maschke): infinite loop on resetting the trigger state.
    //m_trgNewRoi->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    // do nothing if we can not get
    //WPosition crossHairPos = WKernel::getRunningKernel()->getSelectionManager()->getCrosshair()->getPosition();
    WPosition minROIPos( -10.0, -10.0, -10.0 );
    WPosition maxROIPos( 10.0, 10.0, 10.0 );

    osg::ref_ptr< WLROIBox > newRoi = osg::ref_ptr< WLROIBox >( new WLROIBox( minROIPos, maxROIPos, getWidget() ) );
    WKernel::getRunningKernel()->getRoiManager()->addRoi( newRoi );
}

void WLEMDDrawable3DSource::drawCoords()
{
    m_shapeX = new osg::ShapeDrawable;
    m_shapeX->setShape( new osg::Box( osg::Vec3( 200.0, 0.0, 0.0 ), 400, 1, 1 ) ); // X
    m_shapeX->setColor( osg::Vec4( 1.0, 0.0, 0.0, 1.0 ) ); // red

    m_shapeY = new osg::ShapeDrawable;
    m_shapeY->setShape( new osg::Box( osg::Vec3( 0.0, 200.0, 0.0 ), 1, 400, 1 ) ); // Y
    m_shapeY->setColor( osg::Vec4( 0.0, 1.0, 0.0, 1.0 ) ); // green

    m_shapeZ = new osg::ShapeDrawable;
    m_shapeZ->setShape( new osg::Box( osg::Vec3( 0.0, 0.0, 200.0 ), 1, 1, 400 ) ); // Z
    m_shapeZ->setColor( osg::Vec4( 0.0, 0.0, 1.0, 1.0 ) ); //blue

    m_coords = new osg::Geode;
    m_coords->addDrawable( m_shapeX.get() );
    m_coords->addDrawable( m_shapeY.get() );
    m_coords->addDrawable( m_shapeZ.get() );
    m_widget->getScene()->addChild( m_coords );
}

void WLEMDDrawable3DSource::osgAddSurface( const std::vector< WPosition >& positions, const std::vector< WVector3i >& faces )
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
