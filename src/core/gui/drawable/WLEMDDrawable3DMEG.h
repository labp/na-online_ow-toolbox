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

#ifndef WLEMDDRAWABLE3DMEG_H_
#define WLEMDDRAWABLE3DMEG_H_

#include <boost/shared_ptr.hpp>

#include <core/gui/WCustomWidget.h>

#include "WLEMDDrawable3D.h"

namespace LaBP
{
    class WLEMDDrawable3DMEG: public LaBP::WLEMDDrawable3D
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

        explicit WLEMDDrawable3DMEG( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable3DMEG();

        void redraw();

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );
    };
} /* namespace LaBP */
#endif  // WLEMDDRAWABLE3DMEG_H_
