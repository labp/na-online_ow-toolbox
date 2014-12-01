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

#ifndef WFTREQUEST_H_
#define WFTREQUEST_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "modules/ftRtClient/ftb/WFtbCommand.h"

typedef FtBufferRequest WFTRequest;

inline std::ostream& operator<<( std::ostream &strm, const WFTRequest &request )
{
    strm << "WFTRequest: ";
    strm << "version=" << request.out()->def->version;
    strm << ", command=" << wftb::CommandType::name( request.out()->def->command );
    strm << ", bufsize=" << request.out()->def->bufsize;
    return strm;
}

#endif  // WFTREQUEST_H_
