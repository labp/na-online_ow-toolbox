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

#include <string>
#include <utility>  // for pair<>

#include <osgGA/GUIActionAdapter>

#include <core/common/WLogger.h>
#include <core/gui/WCustomWidget.h>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "core/gui/events/WLGUIMouseEvent.h"

#include "WLMarkTimePositionHandler.h"

namespace LaBP
{
    std::string WLMarkTimePositionHandler::CLASS = "WLMarkTimePosition";

    WLMarkTimePositionHandler::WLMarkTimePositionHandler( LaBP::WLEMDDrawable2D::SPtr initiator,
                    LaBP::WLEMDDrawable3D::SPtr acceptor, WModuleOutputData< WLEMMCommand >::SPtr output ) :
                    m_initiator( initiator ), m_acceptor( acceptor ), m_output( output )
    {
    }

    WLMarkTimePositionHandler::~WLMarkTimePositionHandler()
    {
    }

    void WLMarkTimePositionHandler::mouseEventOccurred( const WLGUIMouseEvent& e )
    {
        if( e.getEvent() == WLGUIMouseEvent::Event::CLICK_LEFT )
        {
            if( m_initiator->hasData() )
            {
                const float x_pos = e.getOsgEventAdapter().getX();
                std::pair< boost::shared_ptr< WLEMMeasurement >, size_t > data = m_initiator->getSelectedData( x_pos );
                wlog::debug( CLASS ) << "called handle with pixels: " << x_pos << " and time: " << data.second;
                m_initiator->setSelectedPixel( x_pos );

                m_acceptor->setSelectedSample( data.second );
                m_acceptor->draw( data.first );

                WLEMMCommand::SPtr emmCmd( new WLEMMCommand( WLEMMCommand::Command::TIME_UPDATE ) );
                WLEMMCommand::ParamT param = m_initiator->getSelectedTime();
                emmCmd->setParameter( param );
                m_output->updateData( emmCmd );
            }
        }
    }
} /* namespace LaBP */
