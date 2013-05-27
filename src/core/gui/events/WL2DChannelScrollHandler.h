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

#ifndef WL2DCHANNELSCROLLHANDLER_H_
#define WL2DCHANNELSCROLLHANDLER_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/gui/drawable/WLEMDDrawable2DMultiChannel.h"

#include "WLGUIMouseEventListener.h"

class WL2DChannelScrollHandler: public WLGUIMouseEventListener
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WL2DChannelScrollHandler > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WL2DChannelScrollHandler > ConstSPtr;

    static const std::string CLASS;

    explicit WL2DChannelScrollHandler( LaBP::WLEMDDrawable2DMultiChannel::SPtr initiator );
    virtual ~WL2DChannelScrollHandler();

    virtual void mouseEventOccurred( const WLGUIMouseEvent& e );

private:
    LaBP::WLEMDDrawable2DMultiChannel::SPtr m_initiator;
};

#endif  // WL2DCHANNELSCROLLHANDLER_H_
