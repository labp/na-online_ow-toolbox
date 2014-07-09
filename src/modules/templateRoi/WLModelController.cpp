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

#include <osg/Matrix>

#include "modules/templateRoi/WLModelController.h"

WLModelController::WLModelController( osg::MatrixTransform *node ) :
                m_model( node )
{
}

bool WLModelController::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
{
    if( !m_model )
    {
        return false;
    }

    osg::Matrix matrix = m_model->getMatrix();
    switch( ea.getEventType() )
    {
        case osgGA::GUIEventAdapter::KEYDOWN:
            switch( ea.getKey() )
            {
                case 'a':
                case 'A':
                    matrix *= osg::Matrix::rotate( -0.1f, osg::Z_AXIS );
                    break;
                case 'd':
                case 'D':
                    matrix *= osg::Matrix::rotate( 0.1f, osg::Z_AXIS );
                    break;
                case 'w':
                case 'W':
                    matrix *= osg::Matrix::rotate( -0.1f, osg::X_AXIS );
                    break;
                case 's':
                case 'S':
                    matrix *= osg::Matrix::rotate( 0.1f, osg::X_AXIS );
                    break;
                default:
                    break;
            }
            m_model->setMatrix( matrix );
            break;
        default:
            break;
    }
    return false;

}
