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

#ifndef WFTACHUNKFACTORY_H_
#define WFTACHUNKFACTORY_H_

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

/**
 * WFTAChunkFactory is a generic abstract factory to map classes representing the different FieldTrip chunks with their specific methods and
 * behaviors to the appropriate WLEFTChunkType enum value.
 */
template< typename Enum, typename Base >
class WFTAChunkFactory
{
public:
    /**
     * A shared pointer on a WFTAChunkFactory.
     */
    typedef boost::shared_ptr< WFTAChunkFactory< Enum, Base > > SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Destroys the WFTAChunkFactory.
     */
    virtual ~WFTAChunkFactory();

    /**
     * Commands the derived class to create a new instance.
     *
     * \param e The enum value.
     * \return Returns a pointer on the new instance.
     */
    static boost::shared_ptr< Base > create( Enum e, const char* data, const size_t size );

protected:
    /**
     * Creates a new instance of the derived class.
     *
     * \return Returns a pointer to the new instance.
     */
    virtual boost::shared_ptr< Base > create( const char* data, const size_t size ) = 0;

    /**
     * Gets a static map containing the created @Enum - @Base instances.
     *
     * \return Returns a static map containing the created @Enum - @Base instances.
     */
    static std::map< Enum, WFTAChunkFactory< Enum, Base >* >& lookup();
};

template< typename Enum, typename Base >
const std::string WFTAChunkFactory< Enum, Base >::CLASS = "WFTAChunkFactory";

template< typename Enum, typename Base >
inline WFTAChunkFactory< Enum, Base >::~WFTAChunkFactory()
{
}

template< typename Enum, typename Base >
inline boost::shared_ptr< Base > WFTAChunkFactory< Enum, Base >::create( Enum e, const char* data, const size_t size )
{
    typename std::map< Enum, WFTAChunkFactory< Enum, Base >* >::const_iterator const it = lookup().find( e );
    if( it == lookup().end() )
        return boost::shared_ptr< Base >();

    return it->second->create( data, size );
}

template< typename Enum, typename Base >
inline std::map< Enum, WFTAChunkFactory< Enum, Base >* >& WFTAChunkFactory< Enum, Base >::lookup()
{
    static std::map< Enum, WFTAChunkFactory< Enum, Base >* > l;

    return l;
}

#endif  // WFTACHUNKFACTORY_H_
