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

#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include <osg/Array>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/WLogger.h>
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

void WLEMDDrawable3DSource::setROISelector(
                boost::shared_ptr< WLROISelector< boost::spirit::hold_any, boost::spirit::hold_any > > roiSelector )
{
    m_roiSelecor = roiSelector;
}

boost::shared_ptr< WLROISelector< boost::spirit::hold_any, boost::spirit::hold_any > > WLEMDDrawable3DSource::getROISelector()
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
