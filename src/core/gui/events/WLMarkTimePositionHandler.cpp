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

#include <string>
#include <utility>  // for pair<>

#include <core/common/WException.h>
#include <core/common/WLogger.h>

#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"

#include "WLMarkTimePositionHandler.h"

const std::string WLMarkTimePositionHandler::CLASS = "WLMarkTimePosition";

WLMarkTimePositionHandler::WLMarkTimePositionHandler( WLEMDDrawable2D::SPtr initiator, WLEMDDrawable3D::SPtr acceptor,
                WModuleOutputData< WLEMMCommand >::SPtr output ) :
                WUIViewEventHandler( initiator->getWidget() ), m_initiator( initiator ), m_acceptor( acceptor ), m_output(
                                output )
{
    m_preselection |= GUIEvents::LEFT_MOUSE_BUTTON;
    m_preselection |= GUIEvents::DRAG;
}

WLMarkTimePositionHandler::~WLMarkTimePositionHandler()
{
}

void WLMarkTimePositionHandler::setDrawables( WLEMDDrawable2D::SPtr drawable2D, WLEMDDrawable3D::SPtr drawable3D )
{
    m_initiator = drawable2D;
    m_acceptor = drawable3D;
}

void WLMarkTimePositionHandler::handleDrag( WVector2f mousePos, int buttonMask )
{
    if( buttonMask != GUIEvents::LEFT_MOUSE_BUTTON || !m_initiator->hasData() )
    {
        return;
    }

    const float x_pos = mousePos.x();
    try
    {
        std::pair< WLEMMeasurement::SPtr, size_t > data = m_initiator->getSelectedData( x_pos );
        wlog::debug( CLASS ) << "called handle with pixels: " << x_pos << " and time: " << data.second;
        m_initiator->setSelectedPixel( x_pos );

        m_acceptor->setSelectedSample( data.second );
        m_acceptor->draw( data.first );
    }
    catch( const WException& e )
    {
        wlog::warn( CLASS ) << e.what();
    }
}

void WLMarkTimePositionHandler::handlePush( WVector2f mousePos, int button )
{
    if( button != GUIEvents::LEFT_MOUSE_BUTTON || !m_initiator->hasData() )
    {
        return;
    }

    const float x_pos = mousePos.x();
    try
    {
        std::pair< WLEMMeasurement::SPtr, size_t > data = m_initiator->getSelectedData( x_pos );
        wlog::debug( CLASS ) << "called handle with pixels: " << x_pos << " and time: " << data.second;
        m_initiator->setSelectedPixel( x_pos );

        m_acceptor->setSelectedSample( data.second );
        m_acceptor->draw( data.first );
    }
    catch( const WException& e )
    {
        wlog::warn( CLASS ) << e.what();
        return;
    }

    WLEMMCommand::SPtr emmCmd( new WLEMMCommand( WLEMMCommand::Command::TIME_UPDATE ) );
    WLEMMCommand::ParamT param = m_initiator->getSelectedTime();
    emmCmd->setParameter( param );
    m_output->updateData( emmCmd );
}
