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

#ifndef WLEMDDRAWABLE3DEMPTY_H_
#define WLEMDDRAWABLE3DEMPTY_H_

#include <boost/shared_ptr.hpp>
#include <osg/Geode>
#include <osg/ref_ptr>

#include <core/gui/WCustomWidget.h>

#include "core/data/WLEMMeasurement.h"

#include "WLEMDDrawable3D.h"

namespace LaBP
{
    class WLEMDDrawable3DEmpty: public LaBP::WLEMDDrawable3D
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable3DEmpty > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable3DEmpty > ConstSPtr;

        explicit WLEMDDrawable3DEmpty( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable3DEmpty();

        virtual void draw( WLEMMeasurement::SPtr emm );

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

    private:
        osg::ref_ptr< osg::Geode > m_textGeode;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE3DEMPTY_H_
