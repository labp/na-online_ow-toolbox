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

#ifndef WFTREQUEST_H_
#define WFTREQUEST_H_

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

class WFTRequest: protected FtBufferRequest
{
public:

    typedef boost::shared_ptr< WFTRequest > SPtr;

    typedef messagedef_t WFTMessageDefT;

    typedef message_t WFTMessageT;

    WFTRequest();

    WFTRequest( const WFTMessageT *msg );

    WFTMessageDefT *getMessageDef();

    WFTMessageT *getMessage();

    SimpleStorage *getBuffer();

    FtBufferRequest::out;

};

#endif /* WFTREQUEST_H_ */
