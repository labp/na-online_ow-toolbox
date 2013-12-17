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

#ifndef PACKETIZEREMM_H_
#define PACKETIZEREMM_H_

#include <string>

#include "core/data/WLEMMeasurement.h"

#include "WPacketizer.h"

class WPacketizerEMM: public WPacketizer< WLEMMeasurement >
{
public:
    static const std::string CLASS;

    WPacketizerEMM( WLEMMeasurement::ConstSPtr data, size_t blockSize );
    virtual ~WPacketizerEMM();

    virtual bool hasNext() const;

    virtual WLEMMeasurement::SPtr next();

private:
    std::vector< WLEMData::ConstSPtr > m_emds;
    boost::shared_ptr< WLEMMeasurement::EDataT > m_events;

    size_t m_blockCount;
    bool m_hasData;
};

#endif  // PACKETIZEREMM_H_
