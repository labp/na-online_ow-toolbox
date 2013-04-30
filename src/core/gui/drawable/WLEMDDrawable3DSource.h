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

#ifndef WLEMDDRAWABLE3DSOURCE_H_
#define WLEMDDRAWABLE3DSOURCE_H_

#include <boost/shared_ptr.hpp>

#include <core/gui/WCustomWidget.h>

#include "core/data/WLMatrixTypes.h"

#include "WLEMDDrawable3D.h"

namespace LaBP
{
    class WLEMDDrawable3DSource: public LaBP::WLEMDDrawable3D
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

        explicit WLEMDDrawable3DSource( WCustomWidget::SPtr widget );

        virtual ~WLEMDDrawable3DSource();

        void redraw();

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

    private:
        void osgUpdateSurfaceColor( const MatrixT& data );
    };
} /* namespace LaBP */
#endif  // WLEMDDRAWABLE3DSOURCE_H_
