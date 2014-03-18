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

#ifndef WFTDATA_H_
#define WFTDATA_H_

#include <boost/shared_ptr.hpp>

#include <SimpleStorage.h>

#include "WFTRequestableObject.h"

class WFTData: public WFTRequestableObject
{
public:

    typedef boost::shared_ptr< WFTData > SPtr;

    WFTData();

    WFTData( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType );

    WFTRequest::SPtr asRequest();

    bool parseResponse( WFTResponse::SPtr );

    UINT32_T getSize() const;

    WFTDataDefT& getDataDef();

    void *getData();

protected:

    WFTDataDefT m_def;

    SimpleStorage m_buf;

};

#endif /* WFTDATA_H_ */
