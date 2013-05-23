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

#ifndef WLMODULEINPUTDATARINGBUFFER_H
#define WLMODULEINPUTDATARINGBUFFER_H

#include <cassert>
#include <cstddef> // ptrdiff_t
#include <cstdlib> // calloc
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "WLModuleInputDataCollection.h"

namespace LaBP
{
    /**
     * A ring buffer implementation of WModuleInputDataCollection. It can be used as a producer-consumer-FIFO.
     * Thread safe for 1 producer thread (addData()) and 1 consumer thread (getData()).
     */
    template< typename T >
    class WLModuleInputDataRingBuffer: public WLModuleInputDataCollection< T >
    {
    public:
        /**
         * Shared pointer to this class.
         */
        typedef boost::shared_ptr< WLModuleInputDataRingBuffer > SPtr;

        /**
         * Const shared pointer to this class.
         */
        typedef boost::shared_ptr< const WLModuleInputDataRingBuffer > ConstSPtr;

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
        WLModuleInputDataRingBuffer( size_t capacity, boost::shared_ptr< WModule > module, std::string name = "",
                        std::string description = "" );

        /**
         * Destructor.
         */
        ~WLModuleInputDataRingBuffer();

        /**
         * Returns the oldest element of the buffer and removes this element.
         *
         * \param reset reset the flag of updated() if true (default).
         * \return the oldest element.
         */
        const boost::shared_ptr< T > getData( bool reset = true ) throw( char const* );

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
        size_t capacity();

        /**
         * Removes all elements in the buffer.
         */
        void clear();

        /**
         * Checks whether the collection is empty or not.
         *
         * \return true if this collection contains no elements.
         */
        bool isEmpty();

        /**
         * Returns the current number of elements in this collection.
         *
         * \return current number of elements in this collection.
         */
        size_t size();

    protected:
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
        boost::shared_mutex m_clearLock;
    };
}

template< typename T > const std::string LaBP::WLModuleInputDataRingBuffer< T >::CLASS = "WLModuleInputDataRingBuffer";

template< typename T > const size_t LaBP::WLModuleInputDataRingBuffer< T >::MIN_BUFFER_SIZE = 2;

template< typename T >
LaBP::WLModuleInputDataRingBuffer< T >::WLModuleInputDataRingBuffer( size_t capacity, boost::shared_ptr< WModule > module,
                std::string name, std::string description ) :
                LaBP::WLModuleInputDataCollection< T >( module, name, description )
{
    m_capacity = capacity + 1 < MIN_BUFFER_SIZE ? MIN_BUFFER_SIZE : capacity + 1;
    m_read = 0;
    m_write = 0;
    m_data = ( boost::shared_ptr< T >* )calloc( m_capacity, sizeof( boost::shared_ptr< T > ) );
    assert( m_data );
}

template< typename T >
LaBP::WLModuleInputDataRingBuffer< T >::~WLModuleInputDataRingBuffer()
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
size_t LaBP::WLModuleInputDataRingBuffer< T >::capacity()
{
    return m_capacity - 1;
}

template< typename T >
size_t LaBP::WLModuleInputDataRingBuffer< T >::size()
{
    boost::unique_lock< boost::shared_mutex > exLock( m_clearLock );
    size_t read = m_read;
    size_t write = m_write;
    ptrdiff_t count = write - read;
    if( count < 0 )
        count = m_capacity + count; // NOTE: y - (-x) = y + x
    exLock.unlock();
    return ( size_t )count;
}

template< typename T >
const boost::shared_ptr< T > LaBP::WLModuleInputDataRingBuffer< T >::getData( bool reset ) throw( char const* )
{
#ifdef DEBUG
    wlog::debug( CLASS ) << "getData() called!";
#endif // DEBUG
    if( reset )
    {
        WLModuleInputDataCollection< T >::handledUpdate();
    }

    boost::shared_lock< boost::shared_mutex > shLock( m_clearLock );
    if( m_read == m_write )
    {
        wlog::debug( CLASS ) << "getData(): m_read == m_write";
        throw "No data!";
    }

    size_t nextElement = ( m_read + 1 ) % m_capacity;
    assert( m_data[m_read] );
    boost::shared_ptr< T > element = m_data[m_read];
    assert( element );
    m_data[m_read].reset(); // Prevent memory leak. Not in OW patch request.
    m_read = nextElement;
    shLock.unlock();

    return element;
}

template< typename T >
bool LaBP::WLModuleInputDataRingBuffer< T >::addData( boost::shared_ptr< T > value )
{
    bool rc = false;
#ifdef DEBUG
    wlog::debug( CLASS ) << "addData() called!";
#endif // DEBUG
    boost::shared_lock< boost::shared_mutex > shLock( m_clearLock );
    size_t nextElement = ( m_write + 1 ) % m_capacity;
    if( nextElement != m_read )
    {
        assert( value );
        m_data[m_write] = value;
        assert( m_data[m_write] );
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
void LaBP::WLModuleInputDataRingBuffer< T >::clear()
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
    m_data = ( boost::shared_ptr< T >* )calloc( m_capacity, sizeof(boost::shared_ptr< T >) );
    assert( m_data );

    m_read = 0;
    m_write = 0;

    exLock.unlock();
}

template< typename T >
bool LaBP::WLModuleInputDataRingBuffer< T >::isEmpty()
{
    return size() == 0;
}

#endif  // WLMODULEINPUTDATARINGBUFFER_H
