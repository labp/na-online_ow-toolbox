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

#include "WLEMMCommand.h"

const std::string WLEMMCommand::CLASS = "WLEMMCommand";
// prototype instance as singleton
WLEMMCommand::SPtr WLEMMCommand::m_prototype = WLEMMCommand::SPtr();

WLEMMCommand::WLEMMCommand( Command::Enum command ) :
                m_command( command )
{
}

WLEMMCommand::WLEMMCommand( const WLEMMCommand& o )
{
    m_command = o.m_command;
}

WLEMMCommand::~WLEMMCommand()
{
}

WLEMMCommand::SPtr WLEMMCommand::instance( Command::Enum command )
{
    WLEMMCommand::SPtr instance( new WLEMMCommand( command ) );
    return instance;
}

const std::string WLEMMCommand::getName() const
{
    return "WLEMMCommand";
}

const std::string WLEMMCommand::getDescription() const
{
    return "LaBP packet to transfer commands and data between modules.";
}

boost::shared_ptr< WPrototyped > WLEMMCommand::getPrototype()
{
    if( !m_prototype )
    {
        m_prototype = WLEMMCommand::SPtr( new WLEMMCommand() );
    }

    return m_prototype;
}

WLEMMCommand::Command::Enum WLEMMCommand::getCommand() const
{
    return m_command;
}

void WLEMMCommand::setCommand( Command::Enum command )
{
    m_command = command;
}

WLEMMCommand::MiscCommandT WLEMMCommand::getMiscCommand() const
{
    return m_miscCommand;
}

void WLEMMCommand::setMiscCommand( MiscCommandT param )
{
    m_miscCommand = param;
}

WLEMMeasurement::ConstSPtr WLEMMCommand::getEmm() const
{
    return m_emm;
}

WLEMMeasurement::SPtr WLEMMCommand::getEmm()
{
    return m_emm;
}

void WLEMMCommand::setEmm( WLEMMeasurement::SPtr emm )
{
    m_emm = emm;
}

bool WLEMMCommand::hasEmm() const
{
    return m_emm;
}

const WLEMMCommand::ParamT& WLEMMCommand::getParameter() const
{
    return m_param;
}

void WLEMMCommand::setParameter( ParamT param )
{
    m_param = param;
}
