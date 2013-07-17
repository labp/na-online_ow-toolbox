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

#ifndef WLMARKTIMEPOSITIONHANDLER_H_
#define WLMARKTIMEPOSITIONHANDLER_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModuleOutputData.h>

#include "core/data/WLEMMCommand.h"
#include "core/gui/events/WLGUIMouseEvent.h"
#include "core/gui/events/WLGUIMouseEventListener.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"

namespace LaBP
{
    class WLMarkTimePositionHandler: public WLGUIMouseEventListener
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLMarkTimePositionHandler > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLMarkTimePositionHandler > ConstSPtr;

        static std::string CLASS;

        WLMarkTimePositionHandler( LaBP::WLEMDDrawable2D::SPtr initiator, LaBP::WLEMDDrawable3D::SPtr acceptor,
                        WModuleOutputData< WLEMMCommand >::SPtr output );
        virtual ~WLMarkTimePositionHandler();
        virtual void mouseEventOccurred( const WLGUIMouseEvent& e );

    private:
        LaBP::WLEMDDrawable2D::SPtr m_initiator;
        LaBP::WLEMDDrawable3D::SPtr m_acceptor;
        WModuleOutputData< WLEMMCommand >::SPtr m_output;
    };

} /* namespace LaBP */
#endif  // WLMARKTIMEPOSITIONHANDLER_H_
