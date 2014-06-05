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

#ifndef WFTREQUEST_GETDATA_H_
#define WFTREQUEST_GETDATA_H_

#include <ostream>

#include "WFTRequest.h"

class WFTRequest_GetData: public WFTRequest
{
public:

    /**
     * Constructs a new WFTRequest_GetData.
     *
     * @param begsample The index of the first sample.
     * @param endsample The indes of the last sample.
     */
    WFTRequest_GetData( UINT32_T begsample, UINT32_T endsample );

    /**
     * Destroys the WFTRequest_GetData.
     */
    virtual ~WFTRequest_GetData();

    /**
     * Gets the index of the first sample.
     *
     * @return Returns a 32 bit unsigned integer.
     */
    UINT32_T getBegSample() const;

    /**
     * Gets the index of the last sample.
     *
     * @return Returns a 32 bit unsigned integer.
     */
    UINT32_T getEndSample() const;

private:

    /**
     * The index of the first sample.
     */
    UINT32_T m_begSample;

    /**
     * The index of the last sample.
     */
    UINT32_T m_endSample;
};

inline std::ostream& operator<<( std::ostream& str, const WFTRequest_GetData& request )
{
    str << "WFTRequest_GetData:";
    str << " begSample = " << request.getBegSample();
    str << ", endSample = " << request.getEndSample();

    return str;
}

#endif /* WFTREQUEST_GETDATA_H_ */
