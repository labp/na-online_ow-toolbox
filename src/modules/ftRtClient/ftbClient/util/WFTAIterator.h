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

#ifndef WFTAITERATOR_H_
#define WFTAITERATOR_H_

#include <boost/shared_ptr.hpp>

#include <SimpleStorage.h>

/**
 * The WFTAIterator represents an abstract iterator on a defined memory area.
 * Inherited classes have to implement the methods hasNext() and getNext() for the specific type.
 */
template< typename T >
class WFTAIterator
{
public:
    /**
     * Constructs a new WFTAIterator.
     *
     * \param buf The memory area to iterate.
     * \param size The size of the memory area.
     */
    WFTAIterator( SimpleStorage* const buf, int size );

    /**
     * Destroys the WFTAIterator.
     */
    virtual ~WFTAIterator();

    /**
     * Indicates whether more data objects exist in the storage.
     *
     * \return Returns true if there is another data object, else false.
     */
    virtual bool hasNext() const = 0;

    /**
     * Resets the iterator to the beginning of the data storage.
     */
    virtual void reset();

    /**
     * Returns the next data object in the storage as shared pointer.
     *
     * \return The next data object.
     */
    virtual boost::shared_ptr< T > getNext() = 0;

protected:
    /**
     * A pointer on a memory area containing the data after initializing.
     */
    SimpleStorage* const m_store;

    /**
     * Defines the iterators position inside of the data storage.
     */
    const int m_size;

    /**
     * Contains the memory size of the data storage.
     */
    int m_pos;
};

template< typename T >
inline WFTAIterator< T >::WFTAIterator( SimpleStorage* const buf, int size ) :
                m_store( buf ), m_size( size ), m_pos( 0 )
{
}

template< typename T >
inline WFTAIterator< T >::~WFTAIterator()
{
}

template< typename T >
inline void WFTAIterator< T >::reset()
{
    m_pos = 0;
}

#endif  // WFTAITERATOR_H_
