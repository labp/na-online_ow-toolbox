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

#include "WFTHeader.h"

WFTHeader::WFTHeader()
{
    m_header.def = new WFTHeaderDefT;
    m_header.buf = m_chunkBuffer.data();
}

WFTHeader::~WFTHeader()
{

}

bool WFTHeader::hasChunks()
{
    return m_header.def->bufsize > 0;
}

WFTHeader::WFTHeaderDefT& WFTHeader::getHeaderDef()
{
    return *m_header.def;
}

WFTRequest::SPtr WFTHeader::asRequest()
{
    return WFTRequest::SPtr();
}

bool WFTHeader::parseResponse( WFTResponse::SPtr response )
{
    return response->checkGetHeader( *m_header.def, &m_chunkBuffer );
}

