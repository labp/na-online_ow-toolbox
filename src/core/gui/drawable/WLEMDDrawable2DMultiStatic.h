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

#ifndef WLEMDDRAWABLE2DMULTISTATIC_H_
#define WLEMDDRAWABLE2DMULTISTATIC_H_

#include <string>
#include <utility>  // for pair <>

#include "WLEMDDrawable2DMultiChannel.h"

namespace LaBP
{
    class WLEMDDrawable2DMultiStatic: public LaBP::WLEMDDrawable2DMultiChannel
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable2DMultiStatic > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable2DMultiStatic > ConstSPtr;

        static const std::string CLASS;

        explicit WLEMDDrawable2DMultiStatic( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DMultiStatic();

        virtual void draw( WLEMMeasurement::SPtr emm );

        virtual bool hasData() const;

        virtual std::pair< WLEMMeasurement::SPtr, size_t > getSelectedData( ValueT pixel ) const;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        virtual void osgAddChannels( const WLEMData& emd );

        WLEMMeasurement::SPtr m_emm;
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DMULTISTATIC_H_
