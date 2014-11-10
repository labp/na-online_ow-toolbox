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

#include "WFTResponse.h"

#include <string>


const std::string WFTResponse::CLASS = "WFTResponse";

bool WFTResponse::isValid() const
{
    if( m_response == NULL )
        return false;
    if( m_response->def == NULL )
        return false;
    if( m_response->def->version != VERSION )
        return false;

    return true;
}

bool WFTResponse::hasData() const
{
    if( !isValid() )
    {
        return false;
    }

    return m_response->def->bufsize > 0;
}

const wftb::MessageT WFTResponse::getMessage() const
{
    return *m_response;
}
