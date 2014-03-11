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

#ifndef WFTHEADER_H_
#define WFTHEADER_H_

#include <iostream>
#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <message.h>
#include <SimpleStorage.h>

#include "WFTObject.h"

class WFTHeader: public WFTObject
{

public:

    typedef boost::shared_ptr< WFTHeader > SPtr;

    typedef header_t WFTHeaderT;

    typedef headerdef_t WFTHeaderDefT;

    WFTHeader();

    ~WFTHeader();

    WFTHeaderDefT& getHeaderDef();

    WFTRequest::SPtr asRequest();

    bool parseResponse( WFTResponse::SPtr response );

    bool hasChunks();

private:

    SimpleStorage m_chunkBuffer;

    WFTHeaderT m_header;

};

#endif /* WFTHEADER_H_ */
