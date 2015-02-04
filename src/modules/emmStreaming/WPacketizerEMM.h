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

#ifndef WPACKETIZEREMM_H_
#define WPACKETIZEREMM_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "../emmStreaming/WPacketizer.h"
#include "core/data/WLEMMeasurement.h"


class WPacketizerEMM: public WPacketizer< WLEMMeasurement >
{
public:
    static const std::string CLASS;

    WPacketizerEMM( WLEMMeasurement::ConstSPtr data, WLTimeT blockSize );
    virtual ~WPacketizerEMM();

    virtual bool hasNext() const;

    virtual WLEMMeasurement::SPtr next();

private:
    std::vector< WLEMData::ConstSPtr > m_emds;
    boost::shared_ptr< WLEMMeasurement::EDataT > m_events;

    const WLTimeT m_blockSize;
    size_t m_blockCount;
    bool m_hasData;
};

#endif  // WPACKETIZEREMM_H_
