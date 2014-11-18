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

#include "WFtbCommand.h"

std::string wftb::CommandType::name( command_type_t type )
{
    switch( type )
    {
        case PUT_HDR_:
            return "PUT_HDR";
        case PUT_DAT_:
            return "PUT_DAT";
        case PUT_EVT_:
            return "PUT_EVT";
        case PUT_OK_:
            return "PUT_OK";
        case PUT_ERR_:
            return "PUT_ERR";
        case GET_HDR_:
            return "GET_HDR";
        case GET_DAT_:
            return "GET_DAT";
        case GET_EVT_:
            return "GET_EVT";
        case GET_OK :
            return "GET_OK";
        case GET_ERR_:
            return "GET_ERR";
        case FLUSH_HDR_:
            return "FLUSH_HDR";
        case FLUSH_DAT_:
            return "FLUSH_DAT";
        case FLUSH_EVT_:
            return "FLUSH_EVT";
        case FLUSH_OK_:
            return "FLUSH_OK";
        case FLUSH_ERR_:
            return "FLUSH_ERR";
        case WAIT_DAT_:
            return "WAIT_DAT";
        case WAIT_OK_:
            return "WAIT_OK";
        case WAIT_ERR_:
            return "WAIT_ERR";
        default:
            return "Unknown!";
    }
}
