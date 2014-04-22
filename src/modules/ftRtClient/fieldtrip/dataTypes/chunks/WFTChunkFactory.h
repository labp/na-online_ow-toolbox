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

#ifndef WFTCHUNKFACTORY_H_
#define WFTCHUNKFACTORY_H_

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/fieldtrip/dataTypes/enum/WLEFTChunkType.h"
#include "WFTChannelNames.h"
#include "WFTAChunkFactory.h"
#include "WFTChunk.h"

/**
 * The WFTChunkFactory class creates new instances for the @Derived class type.
 */
template< typename Enum, typename Base, typename Derived >
class WFTChunkFactory: public WFTAChunkFactory< Enum, Base >
{
public:

    /**
     * Constructs a new WFTChunkFactory.
     *
     * @param key The enum value.
     */
    WFTChunkFactory( Enum key );

    /**
     * Destroys the WFTChunkFactory.
     */
    virtual ~WFTChunkFactory();

private:

    /**
     * Creates a new instance of @Derived.
     *
     * @return Returns a pointer on the new instance.
     */
    virtual boost::shared_ptr< Base > create();

    /**
     * The factories position.
     */
    typename std::map< Enum, WFTAChunkFactory< Enum, Base >* >::iterator position;
};

template< typename Enum, typename Base, typename Derived >
inline WFTChunkFactory< Enum, Base, Derived >::WFTChunkFactory( Enum key ) :
                position( this->lookup().insert( std::make_pair< Enum, WFTAChunkFactory< Enum, Base >* >( key, this ) ).first )
{
}

template< typename Enum, typename Base, typename Derived >
inline WFTChunkFactory< Enum, Base, Derived >::~WFTChunkFactory()
{
    this->lookup().erase( position );
}

template< typename Enum, typename Base, typename Derived >
inline boost::shared_ptr< Base > WFTChunkFactory< Enum, Base, Derived >::create()
{
    return boost::shared_ptr< Base >( new Derived() );
}

namespace
{
    //WFTChunkFactory< WLEFTChunkType::Enum, WFTChunk, WFTChannelNames > ChannelNamesFactory( WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES );
}

#endif /* WFTCHUNKFACTORY_H_ */
