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

#ifndef WLMOUSEHANDLER_H_
#define WLMOUSEHANDLER_H_

#include <boost/signals2.hpp>

#include <osg/ref_ptr>

#include <core/graphicsEngine/WPickHandler.h>
#include <core/ui/WUIViewWidget.h>

/**
 *
 */
class WLPickHandler: public WPickHandler
{
public:

    typedef osg::ref_ptr< WLPickHandler > RefPtr;

    static const std::string CLASS;

    WLPickHandler( WUIViewWidget::SPtr widget );

    virtual ~WLPickHandler();

    boost::signals2::signal1< void, WPickInfo >* getPickSignal();

protected:

    virtual void pick( osgViewer::View* view, const osgGA::GUIEventAdapter& ea );

    void updatePickInfoModifierKeys( WPickInfo* pickInfo );

    boost::signals2::signal1< void, WPickInfo > m_pickSignal;

private:

    std::string extractSuitableName( osgUtil::LineSegmentIntersector::Intersections::iterator hitr );

};

#endif /* WLMOUSEHANDLER_H_ */
