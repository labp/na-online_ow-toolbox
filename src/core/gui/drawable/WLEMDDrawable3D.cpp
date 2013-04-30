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

#include <string>
#include <vector>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/LightModel>
#include <osg/ref_ptr>
#include <osg/ShapeDrawable>
#include <osg/Drawable>
#include <osg/Texture>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/dataHandler/exceptions/WDHException.h>
#include <core/graphicsEngine/WTriangleMesh.h>
#include <core/graphicsEngine/WGEUtils.h>
#include <core/graphicsEngine/WGEGeodeUtils.h>

#include "core/dataHandler/WDataSetEMMEnumTypes.h"
#include "core/gui/colorMap/WLColorMap.h"

#include "WLEMDDrawable.h"
#include "WLEMDDrawable3DEEG.h"
#include "WLEMDDrawable3DMEG.h"
#include "WLEMDDrawable3DSource.h"
#include "WLEMDDrawable3DEmpty.h"
#include "WLEMDDrawable3D.h"

namespace LaBP
{
    const std::string WLEMDDrawable3D::CLASS = "WLEMDDrawable3D";

    WLEMDDrawable3D::WLEMDDrawable3D( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable( widget )
    {
        m_selectedSample = -1;
        m_selectedSampleChanged = false;
        m_colorMapChanged = true;
        m_surfaceChanged = true;
        m_state = new osg::StateSet;
        osg::ref_ptr< osg::LightModel > lightModel = new osg::LightModel;
        lightModel->setTwoSided( true );
        m_state->setAttributeAndModes( lightModel );
    }

    WLEMDDrawable3D::~WLEMDDrawable3D()
    {
    }

    WLEMDDrawable3D::SPtr WLEMDDrawable3D::getInstance( WCustomWidget::SPtr widget, LaBP::WEModalityType::Enum modality )
    {
        WLEMDDrawable3D::SPtr drawable3D;
        switch( modality )
        {
            case LaBP::WEModalityType::EEG:
                drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DEEG( widget ) );
                break;
            case LaBP::WEModalityType::MEG:
                // drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DMEG( widget ) );
                drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DEmpty( widget ) );
                break;
            case LaBP::WEModalityType::SOURCE:
                drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DSource( widget ) );
                break;
            default:
                drawable3D = WLEMDDrawable3D::SPtr( new WLEMDDrawable3DEmpty( widget ) );
                break;
        }
        drawable3D->m_modality = modality;
        return drawable3D;
    }

    void WLEMDDrawable3D::draw( LaBP::WDataSetEMM::SPtr emm )
    {

        m_emm = emm;
        m_dataChanged = true;
        redraw();
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

    LaBP::WLColorMap::SPtr WLEMDDrawable3D::getColorMap() const
    {
        return m_colorMap;
    }

    void WLEMDDrawable3D::setColorMap( LaBP::WLColorMap::SPtr colorMap )
    {
        m_colorMap = colorMap;
        m_colorMapChanged = true;
    }

    void WLEMDDrawable3D::osgAddSurface( const std::vector< WPosition >* positions, const std::vector< WVector3i >& faces )
    {
        // draw head surface
        if( m_surfaceChanged )
        {
            m_rootGroup->removeChild( m_surfaceGeode );

            std::size_t nbPositions = positions->size();
            boost::shared_ptr< WTriangleMesh > tri;
            if( faces.size() > 0 )
            {
                osg::ref_ptr< osg::Vec3Array > vertices = wge::osgVec3Array( *positions );
                std::vector< size_t > triangles;

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
                tri = wge::triangulate( *positions, -0.005 );
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

    void WLEMDDrawable3D::clearWidget( bool force )
    {
        if( m_surfaceChanged || force )
        {
            //m_widget->getScene()->remove( m_rootGroup );
        }
    }

    void WLEMDDrawable3D::updateWidget()
    {
        if( m_surfaceChanged )
        {
            //m_widget->getScene()->insert( m_rootGroup );
        }
        m_surfaceChanged = false;
        //m_timeChanged = false;
        m_dataChanged = false;
    }

} /* namespace LaBP */
