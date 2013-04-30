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

#include <boost/shared_ptr.hpp>
#include <osg/Array>

#include <core/gui/WCustomWidget.h>

#include "core/dataHandler/WDataSetEMMSource.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"

#include "WLEMDDrawable3DSource.h"

namespace LaBP
{
    WLEMDDrawable3DSource::WLEMDDrawable3DSource( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable3D( widget )
    {
    }

    WLEMDDrawable3DSource::~WLEMDDrawable3DSource()
    {
    }

    void WLEMDDrawable3DSource::redraw()
    {
        m_draw = true;
    }

    void WLEMDDrawable3DSource::osgUpdateSurfaceColor( const MatrixT& data )
    {
        if( m_selectedSample >= 0 && ( m_dataChanged || m_colorMapChanged ) )
        {
            float color;
            osg::ref_ptr< osg::FloatArray > texCoords =
                            static_cast< osg::FloatArray* >( m_surfaceGeometry->getTexCoordArray( 0 ) );
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
        if( !m_draw )
        {
            return;
        }

        if( !hasData() )
        {
            return;
        }

        LaBP::WLDataSetEMM::ConstSPtr emm = m_emm;
        LaBP::WDataSetEMMSource::ConstSPtr emd = emm->getModality< const WDataSetEMMSource >( WEModalityType::SOURCE );
        LaBP::WLEMMSubject::ConstSPtr subject = emm->getSubject();

        clearWidget();

        if( m_colorMapChanged )
        {
            m_state->setTextureAttributeAndModes( 0, m_colorMap->getAsTexture() );
        }

        osgAddSurface( subject->getSurface( WLEMMSurface::Hemisphere::BOTH ).getVertex().get(),
                        subject->getSurface( WLEMMSurface::Hemisphere::BOTH ).getFaces() );

        osgUpdateSurfaceColor( emd->getMatrix() );

        updateWidget();

        m_draw = false;
        m_colorMapChanged = false;
    }
} /* namespace LaBP */

