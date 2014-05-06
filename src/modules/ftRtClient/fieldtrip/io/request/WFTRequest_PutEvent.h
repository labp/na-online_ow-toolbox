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

#ifndef WFTREQUEST_PUTEVENT_H_
#define WFTREQUEST_PUTEVENT_H_

#include "WFTRequest.h"

class WFTRequest_PutEvent: public WFTRequest
{
public:

    WFTRequest_PutEvent( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type, std::string& value );

    WFTRequest_PutEvent( INT32_T sample, INT32_T offset, INT32_T duration, std::string& type, INT32_T value );

    virtual ~WFTRequest_PutEvent();
};




#endif /* WFTREQUEST_PUTEVENT_H_ */
