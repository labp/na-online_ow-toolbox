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

#ifndef WLPICKINGHANDLER_H_
#define WLPICKINGHANDLER_H_

#include <osg/MatrixTransform>
#include <osg/Node>
#include <osgGA/GUIActionAdapter>
#include <osgGA/GUIEventAdapter>
#include <osgGA/GUIEventHandler>

/*
 *
 */
class WLPickingHandler: public osgGA::GUIEventHandler
{
public:

    static const std::string CLASS;

    virtual bool handle( osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa );

    osg::Node* getOrCreateSelectionBox();

protected:

    osg::ref_ptr< osg::MatrixTransform > m_selectionBox;
};

#endif /* WLPICKINGHANDLER_H_ */
