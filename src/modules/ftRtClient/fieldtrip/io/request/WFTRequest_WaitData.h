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

#ifndef WFTREQUEST_WAITDATA_H_
#define WFTREQUEST_WAITDATA_H_

#include "WFTRequest.h"

class WFTRequest_WaitData: public WFTRequest
{
public:
    WFTRequest_WaitData( UINT32_T nSamples, UINT32_T nEvents, UINT32_T milliseconds );

    virtual ~WFTRequest_WaitData();
};

#endif  // WFTREQUEST_WAITDATA_H_
