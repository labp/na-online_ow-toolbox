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

#include "WFTData.h"

WFTData::WFTData()
{
    m_def.nchans = 0;
    m_def.nsamples = 0;
    m_def.data_type = wftb::DataType::UNKNOWN;
}

WFTData::~WFTData()
{
}

bool WFTData::deserialize( const WFTResponse& response )
{
    m_buf.clear();

    return response.checkGetData( m_def, &m_buf );
}

wftb::bufsize_t WFTData::getSize() const
{
    return m_def.bufsize + sizeof(wftb::DataDefT);
}

wftb::bufsize_t WFTData::getDataSize() const
{
    return m_def.bufsize;
}

const wftb::DataDefT& WFTData::getDataDef() const
{
    return m_def;
}

void* WFTData::getData()
{
    return m_buf.data();
}
