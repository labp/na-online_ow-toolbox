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

#ifndef WLEMDDRAWABLE3DSOURCE_H_
#define WLEMDDRAWABLE3DSOURCE_H_

#include <boost/shared_ptr.hpp>

#include <core/ui/WUIViewWidget.h>

#include "core/data/emd/WLEMData.h"

#include "WLEMDDrawable3D.h"

/**
 * Visualization of the brain surface and the reconstructed activity.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable3DSource: public WLEMDDrawable3D
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3DSource > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3DSource > ConstSPtr;

    explicit WLEMDDrawable3DSource( WUIViewWidget::SPtr widget );

    virtual ~WLEMDDrawable3DSource();

protected:
    virtual void osgNodeCallback( osg::NodeVisitor* nv );

private:
    void osgUpdateSurfaceColor( const WLEMData::DataT& data );
};

#endif  // WLEMDDRAWABLE3DSOURCE_H_
