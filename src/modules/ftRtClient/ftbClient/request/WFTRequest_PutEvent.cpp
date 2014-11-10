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

#include "WFTRequest_PutEvent.h"

#include <string>


WFTRequest_PutEvent::WFTRequest_PutEvent( INT32_T sample, INT32_T offset, INT32_T duration, const std::string& type,
                const std::string& value )
{
    prepPutEvent( sample, offset, duration, type.c_str(), value.c_str() );
}

WFTRequest_PutEvent::WFTRequest_PutEvent( INT32_T sample, INT32_T offset, INT32_T duration, const std::string& type,
                INT32_T value )
{
    prepPutEvent( sample, offset, duration, type.c_str(), value );
}

WFTRequest_PutEvent::~WFTRequest_PutEvent()
{
}

