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

#ifndef WFTOBJECT_H_
#define WFTOBJECT_H_

#include <boost/shared_ptr.hpp>

#include <message.h>

class WFTObject
{
public:

    typedef boost::shared_ptr< WFTObject > SPtr;

    typedef message_t WFTMessageT;

    typedef boost::shared_ptr< const WFTMessageT > WFTMessageT_ConstSPtr;

    typedef messagedef_t WFTMessageDefT;

    typedef header_t WFTHeaderT;

    typedef headerdef_t WFTHeaderDefT;

    typedef data_t WFTDataT;

    typedef datadef_t WFTDataDefT;

    typedef event_t WFTEventT;

    typedef eventdef_t WFTEventDefT;

    typedef waitdef_t WFTWaitDefT;

    typedef ft_chunk_t WFTChunkT;

    typedef boost::shared_ptr< WFTChunkT > WFTChunkT_SPtr;

    typedef ft_chunkdef_t WFTChunkDefT;

    typedef boost::shared_ptr< ft_chunkdef_t > WFTChunkDefT_SPtr;

    virtual ~WFTObject();

    virtual UINT32_T getSize() const = 0;
};

#endif /* WFTOBJECT_H_ */
