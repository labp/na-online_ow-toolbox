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

#ifndef WLEMDDRAWABLE2DSINGLESOURCE_H_
#define WLEMDDRAWABLE2DSINGLESOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/gui/WCustomWidget.h>

#include "core/data/WLEMMeasurement.h"

#include "WLEMDDrawable2DSingleChannel.h"

namespace LaBP
{
    class WLEMDDrawable2DSingleSource: public LaBP::WLEMDDrawable2DSingleChannel
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable2DSingleSource > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable2DSingleSource > ConstSPtr;

        static const std::string CLASS;

        explicit WLEMDDrawable2DSingleSource( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DSingleSource();

        virtual void draw( WLEMMeasurement::SPtr emm );
    };

} /* namespace LaBP */
#endif  // WLEMDDRAWABLE2DSINGLESOURCE_H_
