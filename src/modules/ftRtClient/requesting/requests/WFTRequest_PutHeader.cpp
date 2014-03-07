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

#include "WFTRequest_PutHeader.h"

WFTRequest_PutHeader::WFTRequest_PutHeader( UINT32_T numChannels, UINT32_T dataType, float fsample ) :
                WFTRequest::WFTRequest()
{
    prepPutHeader( numChannels, dataType, fsample );
}

WFTRequest_PutHeader::WFTRequest_PutHeader( UINT16_T version, UINT32_T numChannels, UINT32_T dataType, float fsample )
{
    WFTRequest_PutHeader( numChannels, dataType, fsample );

    setVersion( version );
}

WFTRequest_PutHeader::~WFTRequest_PutHeader()
{

}

