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

#include "WFTResponse.h"

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

const WFTObject::WFTMessageT WFTResponse::getMessage() const
{
    return *m_response;
}
