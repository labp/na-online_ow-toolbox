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

#include "../WFTRequestBuilder.h"
#include "../io/request/WFTRequest_PutData.h"
#include "WFTData.h"

WFTData::WFTData()
{

}

WFTData::WFTData( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType )
{
    m_def.nchans = numChannels;
    m_def.nsamples = numSamples;
    m_def.data_type = dataType;
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

UINT32_T WFTData::getSize() const
{
    return m_def.bufsize + sizeof(WFTDataDefT);
}

WFTDataDefT& WFTData::getDataDef()
{
    return m_def;
}

void *WFTData::getData()
{
    return m_buf.data();
}