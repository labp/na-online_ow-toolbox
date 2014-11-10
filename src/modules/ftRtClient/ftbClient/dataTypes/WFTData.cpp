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

#include "modules/ftRtClient/ftbClient/WFTRequestBuilder.h"
#include "modules/ftRtClient/ftbClient/io/request/WFTRequest_PutData.h"
#include "WFTData.h"

WFTData::WFTData()
{
}

WFTData::WFTData( wftb::nchans_t numChannels, wftb::nsamples_t numSamples, wftb::data_type_t dataType )
{
    m_def.nchans = numChannels;
    m_def.nsamples = numSamples;
    m_def.data_type = dataType;
}

WFTData::~WFTData()
{
}

WFTRequest::SPtr WFTData::asRequest()
{
    WFTRequestBuilder::SPtr builder;
    WFTRequest_PutData::SPtr request = builder->buildRequest_PUT_DAT( m_def.nchans, m_def.nsamples, m_def.data_type,
                    m_buf.data() );

    return WFTRequest_PutData::SPtr( request );
}

bool WFTData::parseResponse( WFTResponse::SPtr response )
{
    m_buf.clear();

    return response->checkGetData( m_def, &m_buf );
}

wftb::bufsize_t WFTData::getSize() const
{
    return m_def.bufsize + sizeof(wftb::DataDefT);
}

wftb::DataDefT& WFTData::getDataDef()
{
    return m_def;
}

void *WFTData::getData()
{
    return m_buf.data();
}
