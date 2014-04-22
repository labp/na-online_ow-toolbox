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

#ifndef WFTACHUNKFACTORY_H_
#define WFTACHUNKFACTORY_H_

#include <map>

using namespace std;

/**
 * WFTAChunkFactory is a generic abstract factory to map classes representing the different FieldTrip chunks with their specific methods and
 * behaviors to the appropriate WLEFTChunkType enum value.
 */
template< typename Enum, typename Base >
class WFTAChunkFactory
{
public:

    /**
     * Destroys the WFTAChunkFactory.
     */
    virtual ~WFTAChunkFactory();

    /**
     * Commands the derived class to create a new instance.
     *
     * @param e The enum value.
     * @return Returns a pointer on the new instance.
     */
    static Base* create( Enum e );

protected:

    /**
     * Gets a static map containing the created @Enum - @Base instances.
     *
     * @return Returns a static map containing the created @Enum - @Base instances.
     */
    static map< Enum, WFTAChunkFactory< Enum, Base >* >& lookup();

private:

    /**
     * Creates a new instance of the derived class.
     *
     * @return Returns a pointer to the new instance.
     */
    virtual Base* create() = 0;
};

template< typename Enum, typename Base >
inline WFTAChunkFactory< Enum, Base >::~WFTAChunkFactory()
{
}

template< typename Enum, typename Base >
inline Base* WFTAChunkFactory< Enum, Base >::create( Enum e )
{
    typename map< Enum, WFTAChunkFactory< Enum, Base >* >::const_iterator const it = lookup().find( e );
    if( it == lookup().end() )
        return 0;

    return it->second->create();
}

template< typename Enum, typename Base >
inline map< Enum, WFTAChunkFactory< Enum, Base >* >& WFTAChunkFactory< Enum, Base >::lookup()
{
    static map< Enum, WFTAChunkFactory< Enum, Base >* > l;

    return l;
}

#endif /* WFTACHUNKFACTORY_H_ */
