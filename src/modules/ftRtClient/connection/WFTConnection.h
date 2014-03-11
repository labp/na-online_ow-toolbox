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

#ifndef WFTCONNECTION_H_
#define WFTCONNECTION_H_

#include <boost/shared_ptr.hpp>

#include <FtBuffer.h>

class WFTConnection: protected FtConnection
{
public:

    typedef boost::shared_ptr< WFTConnection > SPtr;

    WFTConnection( int retry = 0 );

    virtual ~WFTConnection();

    virtual bool connect() = 0;

    virtual bool connect( std::string address );

    bool isOpen() const;

    void disconnect();

    int getSocket() const;
};

#endif /* WFTCONNECTION_H_ */
