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

#include <osg/Array>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Matrix>
#include <osg/PolygonMode>
#include <osg/ShapeDrawable>
#include <osg/StateAttribute>
#include <osg/StateSet>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include <osgViewer/Viewer>

#include <core/common/WLogger.h>

#include <modules/templateRoi/WLPickingHandler.h>

const std::string WLPickingHandler::CLASS = "WLPickingHandler";

bool WLPickingHandler::handle( osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
{
    wlog::debug(CLASS) << "handle()!";

    if( ea.getEventType() != osgGA::GUIEventAdapter::RELEASE || ea.getButton() != osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON
                    || !( ea.getModKeyMask() & osgGA::GUIEventAdapter::MODKEY_CTRL ) )
    {
        return false;
    }

    osgViewer::Viewer* viewer = dynamic_cast< osgViewer::Viewer* >( &aa );
    if( viewer )
    {
        osg::ref_ptr< osgUtil::LineSegmentIntersector > intersector = new osgUtil::LineSegmentIntersector(
                        osgUtil::Intersector::WINDOW, ea.getX(), ea.getY() );
        osgUtil::IntersectionVisitor iv( intersector.get() );
        iv.setTraversalMask( ~0x1 );
        viewer->getCamera()->accept( iv );
        if( intersector->containsIntersections() )
        {
            osgUtil::LineSegmentIntersector::Intersection& result =
                            ( osgUtil::LineSegmentIntersector::Intersection& )*( intersector->getIntersections().begin() );
            osg::BoundingBox bb = result.drawable->getBound();
            osg::Vec3 worldCenter = bb.center() * osg::computeLocalToWorld( result.nodePath );
            m_selectionBox->setMatrix(
                            osg::Matrix::scale( bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin() )
                                            * osg::Matrix::translate( worldCenter ) );
        }
    }
    return false;

}

osg::Node* WLPickingHandler::getOrCreateSelectionBox()
{
    if( !m_selectionBox )
    {
        osg::ref_ptr< osg::Geode > geode = new osg::Geode;
        geode->addDrawable( new osg::ShapeDrawable( new osg::Box( osg::Vec3(), 1.0f ) ) );

        m_selectionBox = new osg::MatrixTransform;
        m_selectionBox->setNodeMask( 0x1 );
        m_selectionBox->addChild( geode.get() );

        osg::StateSet* ss = m_selectionBox->getOrCreateStateSet();
        ss->setMode( GL_LIGHTING, osg::StateAttribute::OFF );
        ss->setAttributeAndModes( new osg::PolygonMode( osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE ) );
    }

    return m_selectionBox.get();
}
