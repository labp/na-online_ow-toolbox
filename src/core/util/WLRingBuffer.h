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

#ifndef WLRINGBUFFER_H
#define WLRINGBUFFER_H

#include <cstddef>  // ptrdiff_t
#include <cstdlib>  // calloc
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/mutex.hpp>

#include <core/common/WAssert.h>
#include <core/common/WException.h>
#include <core/common/WLogger.h>

/**
 * A ring buffer or circular implementation. Template parameter is wrapped in a boost::shared_ptr< T >.
 * It can be used as a producer-consumer-FIFO. Thread safe for 1 producer thread (addData()) and 1 consumer thread (getData()).
 *
 * \author pieloth
 * \ingroup util
 */
template< typename T >
class WLRingBuffer
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLRingBuffer< T > > SPtr;

    static const std::string CLASS;

    static const size_t MIN_BUFFER_SIZE;

    /**
     * Constructor.
     *
     * \param capacity number of elements which should be buffered.
     * \param module the module which is owner of this connector.
     * \param name The name of this connector.
     * \param description Short description of this connector.
     */
    explicit WLRingBuffer( size_t capacity );

    /**
     * Destructor.
     */
    ~WLRingBuffer();

    /**
     * Returns the oldest element of the buffer and removes this element. Throws WException if buffer is empty!
     *
     * \return the oldest element or WException
     */
    boost::shared_ptr< T > popData() throw( WException );

    /**
     * Returns the element relative to the offset and read pointer. Throws WException if requested data is empty or data has been overwritten by producer thread!
     *
     * \return element or WException
     */
    boost::shared_ptr< T > getData( ptrdiff_t offset ) throw( WException );

    /**
     * Adds an elements to the buffer.
     *
     * \param value element whose presence in this collection is to be ensured.
     * \return true if the buffer holds the element. false if the element could not be added e.g. buffer is full.
     */
    bool addData( boost::shared_ptr< T > value );

    /**
     * Buffer size.
     *
     * \return the maximum count of elements which can be stored.
     */
    size_t capacity() const;

    /**
     * Removes all elements in the buffer.
     */
    void clear();

    /**
     * Checks whether the collection is empty or not.
     *
     * \return true if this collection contains no elements.
     */
    bool isEmpty() const;

    /**
     * Returns the current number of new elements in this collection.
     *
     * \return current number of elements in this collection.
     */
    size_t size() const;

    size_t nmod( ptrdiff_t a, size_t n ) const;

private:
    /**
     * Data vector.
     */
    boost::shared_ptr< T >* volatile m_data;

    /**
     * Buffer size.
     */
    size_t m_capacity;

    /**
     * Read pointer.
     */
    volatile size_t m_read;

    /**
     * Write pointer.
     */
    volatile size_t m_write;

    /**
     * Lock to clear the buffer.
     */
    boost::shared_mutex mutable m_clearLock;
};

template< typename T > const std::string WLRingBuffer< T >::CLASS = "WLRingBuffer";

template< typename T > const size_t WLRingBuffer< T >::MIN_BUFFER_SIZE = 2;

template< typename T > inline size_t WLRingBuffer< T >::nmod( ptrdiff_t a, size_t n ) const
{
    WAssertDebug( n > 0, "nmod(a, n): n must be greater than 0!" );
    if( a > -1 )
    {
        return a % n;
    }
    else
    {
        const ptrdiff_t r = labs( a ) % n; // a * x > n for x > 1
        a = ( -r + n ) % n;
        return a;
    }
}

template< typename T >
WLRingBuffer< T >::WLRingBuffer( size_t capacity )
{
    m_capacity = capacity + 1 < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : capacity + 1;
    m_read = 0;
    m_write = 0;
    m_data = ( boost::shared_ptr< T >* )calloc( m_capacity, sizeof( boost::shared_ptr< T > ) );
    WAssert( m_data, "Could not allocate m_data!" );
}

template< typename T >
WLRingBuffer< T >::~WLRingBuffer()
{
    boost::unique_lock< boost::shared_mutex > exLock( m_clearLock );

    for( size_t i = 0; i < m_capacity; ++i )
    {
        if( m_data[i] )
        {
            m_data[i].reset();
        }
    }
    free( m_data );

    exLock.unlock();
}

template< typename T >
size_t WLRingBuffer< T >::capacity() const
{
    return m_capacity - 1;
}

template< typename T >
size_t WLRingBuffer< T >::size() const
{
    boost::unique_lock< boost::shared_mutex > exLock( m_clearLock );
    size_t read = m_read;
    size_t write = m_write;
    ptrdiff_t count = write - read;
    if( count < 0 )
        count = m_capacity + count; // NOTE: y - (-x) = y + x
    exLock.unlock();
    return count;
}

template< typename T >
boost::shared_ptr< T > WLRingBuffer< T >::popData() throw( WException )
{
#ifdef DEBUG
    wlog::debug( CLASS ) << "popData() called!";
#endif // DEBUG
    boost::shared_lock< boost::shared_mutex > shLock( m_clearLock );
    WAssert( m_read != m_write, "Buffer is empty!" );

    size_t nextElement = ( m_read + 1 ) % m_capacity;
    WAssertDebug( m_data[m_read], "Requested data is empty!" );
    boost::shared_ptr< T > element = m_data[m_read];
    WAssertDebug( element, "Requested data is empty!" );
    m_read = nextElement;
    shLock.unlock();

    return element;
}

template< typename T >
boost::shared_ptr< T > WLRingBuffer< T >::getData( ptrdiff_t offset = 0 ) throw( WException )
{
#ifdef DEBUG
    wlog::debug( CLASS ) << "getData() called!";
#endif // DEBUG
    boost::shared_lock< boost::shared_mutex > shLock( m_clearLock );

    const size_t preWrite = m_write;
    const size_t read = WLRingBuffer< T >::nmod( m_read + offset, m_capacity );
    boost::shared_ptr< T > element = m_data[read];
    const size_t postWrite = m_write;

    shLock.unlock();

    WAssert( element, "Requested data is empty!" );

    // Check if requested element was overridden
    bool success = false;
    if( preWrite == postWrite )
    {
        success = true;
    }
    else
        if( postWrite < preWrite )
        {
            success = postWrite <= read && read < preWrite;
        }
        else
            if( preWrite < postWrite )
            {
                success = !( preWrite <= read && read < postWrite );
            }
    WAssert( success, "Data has been overridden during read operation!" );

    return element;
}

template< typename T >
bool WLRingBuffer< T >::addData( boost::shared_ptr< T > value )
{
    bool rc = false;
#ifdef DEBUG
    wlog::debug( CLASS ) << "addData() called!";
#endif // DEBUG
    boost::shared_lock< boost::shared_mutex > shLock( m_clearLock );
    size_t nextElement = ( m_write + 1 ) % m_capacity;
    if( nextElement != m_read )
    {
        WAssertDebug( value, "Element to add is empty!" );
        m_data[m_write] = value;
        WAssertDebug( m_data[m_write], "Could not store element!" );
        m_write = nextElement;
        rc = true;
    }
    else
    {
        wlog::debug( CLASS ) << "addData() Queue is full!";
        rc = false;
    }
    shLock.unlock();
    return rc;
}

template< typename T >
void WLRingBuffer< T >::clear()
{
    wlog::debug( CLASS ) << "clear() called!";
    boost::unique_lock< boost::shared_mutex > exLock( m_clearLock );

    for( size_t i = 0; i < m_capacity; ++i )
    {
        if( m_data[i] )
        {
            m_data[i].reset();
        }
    }

    free( m_data );
    m_data = ( boost::shared_ptr< T >* )calloc( m_capacity, sizeof( boost::shared_ptr< T > ) );
    WAssert( m_data, "Could not allocate m_data!" );

    m_read = 0;
    m_write = 0;

    exLock.unlock();
}

template< typename T >
bool WLRingBuffer< T >::isEmpty() const
{
    return size() == 0;
}

#endif  // WLRINGBUFFER_H
