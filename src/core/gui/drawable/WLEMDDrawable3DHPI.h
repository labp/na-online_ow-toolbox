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

#ifndef WLEMDDRAWABLE3DHPI_H_
#define WLEMDDRAWABLE3DHPI_H_

#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/Matrixd>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>

#include "core/data/WLPositions.h"
#include "WLEMDDrawable3D.h"

/**
 * Visualize the HPI coils in the MEG helmet.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable3DHPI: public WLEMDDrawable3D
{
public:

    /**
     * Enumeration for the user's view on the MEG helmet.
     */
    enum View
    {
        VIEW_TOP, VIEW_SIDE, VIEW_FRONT
    };

    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3DHPI > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3DHPI > ConstSPtr;

    WLEMDDrawable3DHPI( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable3DHPI();

    /**
     * Sets the user view on the MEG helmet.
     *
     * \param view Top, side, front.
     */
    void setView( View view );

protected:
    virtual void osgNodeCallback( osg::NodeVisitor* nv );

private:
    /**
     * Returns a transformation matrix to rotate the MEG helmet.
     *
     * \param view Requested view top, side, front.
     * \return Transformation matrix.
     */
    static osg::Matrixd getTransformation( View view );

    void osgInitMegHelmet();
    void osgAddOrUpdateHpiCoils( const WLPositions& positions );

    osg::Matrixd m_viewTransformation; /**< Transformation matrix for osg::MatrixTransform. */
    osg::ref_ptr< osg::MatrixTransform > m_viewGeode; /**< Transforms the MEG device coordinates for the requested view. */

    osg::ref_ptr< osg::Geode > m_magSensorsGeode;
    osg::ref_ptr< osg::Geode > m_hpiCoilsGeode;
};

#endif  // WLEMDDRAWABLE3DHPI_H_
