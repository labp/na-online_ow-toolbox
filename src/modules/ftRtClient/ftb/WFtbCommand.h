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

#ifndef WFTBCOMMAND_H_
#define WFTBCOMMAND_H_

#include <stdint.h>

#include <message.h>

namespace wftb
{
    typedef uint16_t command_type_t;

    /**
     * Encodes the type of request. Underscore at the end avoids naming conflicts with defines in messages.h
     */
    namespace CommandType
    {
        const command_type_t PUT_HDR_ = PUT_HDR;
        const command_type_t PUT_DAT_ = PUT_DAT;
        const command_type_t PUT_EVT_ = PUT_EVT;
        const command_type_t PUT_OK_ = PUT_OK;
        const command_type_t PUT_ERR_ = PUT_ERR;

        const command_type_t GET_HDR_ = GET_HDR;
        const command_type_t GET_DAT_ = GET_DAT;
        const command_type_t GET_EVT_ = GET_EVT;
        const command_type_t GET_OK_ = GET_OK;
        const command_type_t GET_ERR_ = GET_ERR;

        const command_type_t FLUSH_HDR_ = FLUSH_HDR;
        const command_type_t FLUSH_DAT_ = FLUSH_DAT;
        const command_type_t FLUSH_EVT_ = FLUSH_EVT;
        const command_type_t FLUSH_OK_ = FLUSH_OK;
        const command_type_t FLUSH_ERR_ = FLUSH_ERR;

        const command_type_t WAIT_DAT_ = WAIT_DAT;
        const command_type_t WAIT_OK_ = WAIT_OK;
        const command_type_t WAIT_ERR_ = WAIT_ERR;
    } /* namespace Command */
} /* namespace wftb */
#endif  // WFTBCOMMAND_H_
