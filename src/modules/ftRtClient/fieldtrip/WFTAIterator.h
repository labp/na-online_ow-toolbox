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

#ifndef WFTAITERATOR_H_
#define WFTAITERATOR_H_

#include <boost/shared_ptr.hpp>

#include <message.h>
#include <SimpleStorage.h>

template< typename T >
class WFTAIterator
{
public:

    WFTAIterator( SimpleStorage& buf, UINT32_T size );

    virtual ~WFTAIterator();

    virtual bool hasNext() const = 0;

    virtual void reset() = 0;

    virtual boost::shared_ptr< T > getNext() = 0;

protected:

    SimpleStorage &m_store;

    const UINT32_T m_size;

    UINT32_T m_pos;

};

template< typename T >
inline WFTAIterator< T >::WFTAIterator( SimpleStorage &buf, UINT32_T size ) :
                m_store( buf ), m_size( size ), m_pos( 0 )
{
}

template< typename T >
inline WFTAIterator< T >::~WFTAIterator()
{
}

#endif /* WFTAITERATOR_H_ */
