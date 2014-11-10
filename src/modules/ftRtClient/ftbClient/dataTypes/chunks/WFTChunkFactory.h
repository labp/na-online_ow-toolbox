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

#ifndef WFTCHUNKFACTORY_H_
#define WFTCHUNKFACTORY_H_

#include <map>

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/ftbClient/dataTypes/enum/WLEFTChunkType.h"
#include "WFTChunkChanNames.h"
#include "WFTChunkNeuromagHdr.h"
#include "WFTChunkNeuromagIsotrak.h"
#include "WFTAChunk.h"
#include "WFTAChunkFactory.h"

/**
 * The WFTChunkFactory class creates new instances for the @Derived class type.
 */
template< typename Enum, typename Base, typename Derived >
class WFTChunkFactory: public WFTAChunkFactory< Enum, Base >
{
public:
    /**
     * A shared pointer on a WFTChunkFactory.
     */
    typedef boost::shared_ptr< WFTChunkFactory< Enum, Base, Derived > > SPtr;

    /**
     * Constructs a new WFTChunkFactory.
     *
     * \param key The enum value.
     */
    explicit WFTChunkFactory( Enum key ) :
                    m_position(
                                    this->lookup().insert( std::make_pair< Enum, WFTAChunkFactory< Enum, Base >* >( key, this ) ).first )
    {
    }

    /**
     * Destroys the WFTChunkFactory.
     */
    virtual ~WFTChunkFactory();

protected:
    /**
     * Creates a new instance of @Derived.
     *
     * Inherited method from WFTAChunkFactory.
     *
     * \return Returns a pointer on the new instance.
     */
    virtual boost::shared_ptr< Base > create( const char* data, const size_t size );

private:
    /**
     * The factories position.
     */
    typename std::map< Enum, WFTAChunkFactory< Enum, Base >* >::iterator m_position;
};

template< typename Enum, typename Base, typename Derived >
inline WFTChunkFactory< Enum, Base, Derived >::~WFTChunkFactory()
{
}

template< typename Enum, typename Base, typename Derived >
inline boost::shared_ptr< Base > WFTChunkFactory< Enum, Base, Derived >::create( const char* data, const size_t size )
{
    return boost::dynamic_pointer_cast< Base >( boost::shared_ptr< Derived >( new Derived( data, size ) ) );
}

namespace
{
    WFTChunkFactory< WLEFTChunkType::Enum, WFTAChunk, WFTChunkChanNames > const ChannelNamesFactory(
                        WLEFTChunkType::FT_CHUNK_CHANNEL_NAMES );
    WFTChunkFactory< WLEFTChunkType::Enum, WFTAChunk, WFTChunkNeuromagHdr > const NeuromagHeaderFactory(
                    WLEFTChunkType::FT_CHUNK_NEUROMAG_HEADER );
    WFTChunkFactory< WLEFTChunkType::Enum, WFTAChunk, WFTChunkNeuromagIsotrak > const NeuromagIsotrakFactory(
                    WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK );
}

#endif  // WFTCHUNKFACTORY_H_
