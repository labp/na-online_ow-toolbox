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

#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/ref_ptr>

#include "WLEMDDrawable3D.h"

/**
 * TODO(pieloth): documentation
 *
 * \author pieloth
 */
class WLEMDDrawable3DHPI: public WLEMDDrawable3D
{
public:
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

protected:
    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    void osgInitMegHelmet();

    osg::ref_ptr< osg::Geode > m_magSensorsGeode;
    std::vector< osg::ref_ptr< osg::ShapeDrawable > > m_magSensorsDrawables;

    void osgAddOrUpdateHpiCoils( const std::vector< WPosition >& positions );

    osg::ref_ptr< osg::Geode > m_hpiCoilsGeode;
    std::vector< osg::ref_ptr< osg::ShapeDrawable > > m_hpiCoilsDrawables;
};

#endif  // WLEMDDRAWABLE3DHPI_H_
