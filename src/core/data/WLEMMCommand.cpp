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

#include "WLEMMCommand.h"

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

/**
 * Gets the description for this prototype.
 *
 * \return the description
 */
const std::string WLEMMCommand::getDescription() const
{
    return "LaBP packet to transfer commands and data between modules.";
}

/**
 * Returns a prototype instantiated with the true type of the deriving class.
 *
 * \return the prototype.
 */
boost::shared_ptr< WPrototyped > WLEMMCommand::getPrototype()
{
    if( !m_prototype )
    {
        m_prototype = boost::shared_ptr< WPrototyped >( new WLEMMCommand() );
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