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

#ifndef WLEMDDRAWABLE3DEEG_H_
#define WLEMDDRAWABLE3DEEG_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/ShapeDrawable>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLPositions.h"
#include "WLEMDDrawable3D.h"

/**
 * Visualization of a EEG sensor cap and the mapped the data.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable3DEEG: public WLEMDDrawable3D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3DEEG > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3DEEG > ConstSPtr;

    explicit WLEMDDrawable3DEEG( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable3DEEG();

    void setLabels( bool labelOn );

protected:
    virtual bool mustDraw() const;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

private:
    void osgAddLabels( const WLPositions& positions, const std::vector< std::string >& labels );

    void osgAddNodes( const WLPositions& positions );

    void osgUpdateSurfaceColor( const WLEMData::DataT& data );

    void osgUpdateNodesColor( const WLEMData::DataT& data );

    bool m_electrodesChanged;

    osg::ref_ptr< osg::Geode > m_electrodesGeode;

    std::vector< osg::ref_ptr< osg::ShapeDrawable > > m_electrodesDrawables;

    bool m_labelsChanged;
    bool m_labelsOn;

    osg::ref_ptr< osg::Geode > m_labesGeode;
};

#endif  // WLEMDDRAWABLE3DEEG_H_
