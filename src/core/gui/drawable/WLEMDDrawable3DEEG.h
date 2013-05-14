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

#ifndef WLEMDDRAWABLE3DEEG_H_
#define WLEMDDRAWABLE3DEEG_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/ShapeDrawable>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/gui/WCustomWidget.h>

#include "WLEMDDrawable3D.h"

namespace LaBP
{
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

        explicit WLEMDDrawable3DEEG( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable3DEEG();

        void clearWidget( bool force = false );

    protected:
        virtual bool mustDraw() const;

        virtual void osgNodeCallback( osg::NodeVisitor* nv );

    private:
        void osgAddLabels( const std::vector< WPosition >* positions, const std::vector< std::string >& labels );
        bool m_labelsOn;

        void osgAddNodes( const std::vector< WPosition >* positions );

        void osgUpdateSurfaceColor( const LaBP::WLEMD::DataT& data );

        void osgUpdateNodesColor( const LaBP::WLEMD::DataT& data );

        bool m_electrodesChanged;

        osg::ref_ptr< osg::Geode > m_electrodesGeode;

        std::vector< osg::ref_ptr< osg::ShapeDrawable > > m_electrodesDrawables;

        bool m_labelsChanged;

        osg::ref_ptr< osg::Geode > m_labesGeode;

    };
} /* namespace LaBP */
#endif  // WLEMDDRAWABLE3DEEG_H_
