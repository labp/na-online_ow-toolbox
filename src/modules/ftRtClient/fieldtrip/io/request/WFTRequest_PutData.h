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

#ifndef WFTREQUEST_PUTDATA_H_
#define WFTREQUEST_PUTDATA_H_

#include "WFTRequest.h"

class WFTRequest_PutData: public WFTRequest
{
public:

    WFTRequest_PutData( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType, const void *data );

    virtual ~WFTRequest_PutData();
};

#endif /* WFTREQUEST_PUTDATA_H_ */
