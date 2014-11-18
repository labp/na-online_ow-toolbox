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

#ifndef WFTRESPONSE_H_
#define WFTRESPONSE_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbCommand.h"

/**
 * Wrapper class for a response created through a FieldTrip request.
 */
class WFTResponse: public FtBufferResponse
{
public:
    static const std::string CLASS;

    /**
     * Define a shared pointer on a response.
     */
    typedef boost::shared_ptr< WFTResponse > SPtr;

    /**
     * This method tests the arrived data for mistakes. It should be called before getting any data for the response.
     *
     * \return True if the data are valid, else false.
     */
    bool isValid() const;

    bool hasData() const;
};

inline std::ostream& operator<<( std::ostream &strm, const WFTResponse &response )
{
    strm << WFTResponse::CLASS << ": ";
    strm << "version=" << response.m_response->def->version;
    strm << ", command=" << wftb::CommandType::name( response.m_response->def->command );
    strm << ", bufsize=" << response.m_response->def->bufsize;

    return strm;
}

#endif  // WFTRESPONSE_H_
