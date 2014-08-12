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

#include <core/common/WLogger.h>
#include <core/graphicsEngine/WPickHandler.h>
#include <core/graphicsEngine/WPickInfo.h>

#include "WLROI.h"

const std::string WLROI::CLASS = "WLROI";

WLROI::WLROI( WUIViewWidget::SPtr widget ) :
                WROI(), m_color( osg::Vec4( 0.188f, 0.388f, 0.773f, 0.5f ) ), m_notColor(
                                osg::Vec4( 0.828f, 0.391f, 0.391f, 0.5f ) ), m_widget( widget )
{
    m_viewer = m_widget->getViewer();
    m_pickHandler = m_viewer->getPickHandler();
    //m_pickHandler->getPickSignal()->connect( boost::bind( &WLROI::registerRedrawRequest, this, _1 ) );

    m_mouseHandler = new WLPickHandler( m_widget );
    m_mouseHandler->getPickSignal()->connect( boost::bind( &WLROI::registerRedrawRequest, this, _1 ) );
    m_widget->addEventHandler( m_mouseHandler.get() );
}

WLROI::~WLROI()
{
}

void WLROI::setColor( osg::Vec4 color )
{
    m_color = color;
}

void WLROI::setNotColor( osg::Vec4 notColor )
{
    m_notColor = notColor;
}

void WLROI::registerRedrawRequest( WPickInfo pickInfo )
{
    boost::unique_lock< boost::shared_mutex > lock;
    lock = boost::unique_lock< boost::shared_mutex >( m_updateLock );

    m_pickInfo = pickInfo;

    lock.unlock();
}

void WLROI::updateColor( osg::Vec4 color )
{
    osg::ref_ptr< osg::Vec4Array > colors = osg::ref_ptr< osg::Vec4Array >( new osg::Vec4Array );
    colors->push_back( color );

    WColor outline( 0.0, 0.0, 0.0, 1.0 );
    // NOTE: also add a black color for the solid outlines
    colors->push_back( outline );
    colors->push_back( outline );
    colors->push_back( outline );
    colors->push_back( outline );
    colors->push_back( outline );
    colors->push_back( outline );
    m_surfaceGeometry->setColorArray( colors );
    m_surfaceGeometry->setColorBinding( osg::Geometry::BIND_PER_PRIMITIVE_SET );
}
