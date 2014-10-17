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

#ifndef WLEMDDRAWABLE3DMEG_H_
#define WLEMDDRAWABLE3DMEG_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/ref_ptr>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/enum/WLEModality.h"
#include "WLEMDDrawable3D.h"

class WLEMDDrawable3DMEG: public WLEMDDrawable3D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3DMEG > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3DMEG > ConstSPtr;

    explicit WLEMDDrawable3DMEG( WUIViewWidget::SPtr widget );

    WLEMDDrawable3DMEG( WUIViewWidget::SPtr widget, WLEModality::Enum coilType );

    virtual ~WLEMDDrawable3DMEG();
    void setLabels( bool labelOn );

protected:
    virtual bool mustDraw() const;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

private:
    const WLEModality::Enum m_coilType;

    void osgAddLabels( const std::vector< WPosition >& positions, const std::vector< std::string >& labels );

    void osgAddNodes( const std::vector< WPosition >& positions );

    void osgUpdateSurfaceColor( const WLEMData::DataT& data );

    void osgUpdateNodesColor( const WLEMData::DataT& data );

    bool m_electrodesChanged;

    osg::ref_ptr< osg::Geode > m_electrodesGeode;

    std::vector< osg::ref_ptr< osg::ShapeDrawable > > m_electrodesDrawables;

    bool m_labelsChanged;
    bool m_labelsOn;

    osg::ref_ptr< osg::Geode > m_labesGeode;
};

#endif  // WLEMDDRAWABLE3DMEG_H_
