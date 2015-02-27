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

#ifndef WLRINGBUFFER_TEST_H
#define WLRINGBUFFER_TEST_H

#include "../WLRingBuffer.h"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <string>

#include <cxxtest/TestSuite.h>

#include <core/common/WException.h>
#include <core/common/WLogger.h>


/**
 * Unit tests our WFiber class
 */
class WLRingBufferTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_constructor()
    {
        TS_TRACE( "test_constructor" );
        WLRingBuffer< std::string >::SPtr buffer( new WLRingBuffer< std::string >( 1 ) );
        TS_ASSERT( 0 < buffer->capacity() );

        buffer.reset( new WLRingBuffer< std::string >( 3 ) );
        TS_ASSERT( buffer->capacity() > 2 );
    }

    void test_nmod()
    {
        TS_TRACE( "test_nmod" );

        size_t capacity = 5;
        WLRingBuffer< std::string >::SPtr buffer( new WLRingBuffer< std::string >( capacity ) );
        ptrdiff_t a;
        size_t n, result;

        // 5 % 2 = 1
        a = 5;
        n = 2;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 1, result );

        // 5 % 7 = 5
        a = 5;
        n = 7;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 5, result );

        // 72 % 23 = 3
        a = 72;
        n = 23;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 3, result );

        // -5 % 2 = 1
        a = 5;
        n = 2;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 1, result );

        // -5 % 7 = 2
        a = -5;
        n = 7;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 2, result );

        // -72 % 23 = 20
        a = -72;
        n = 23;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 20, result );

        // -15 % 7 = 6
        a = -15;
        n = 7;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 6, result );

        // -4 % 13 = 9
        a = -4;
        n = 13;
        result = buffer->nmod( a, n );
        TS_ASSERT_EQUALS( 9, result );
    }

    void test_size()
    {
        TS_TRACE( "test_size" );

        size_t capacity = 5;
        WLRingBuffer< std::string >::SPtr buffer( new WLRingBuffer< std::string >( capacity ) );
        TS_ASSERT( buffer->capacity() == capacity );
        boost::shared_ptr< std::string > element;
        for( size_t i = 0; i < capacity; ++i )
        {
            element.reset( new std::string( "foo" ) );
            buffer->addData( element );
            TS_ASSERT( buffer->size() == i+1 );
            TS_ASSERT( !buffer->isEmpty() );
        }

        buffer->clear();
        TS_ASSERT( buffer->isEmpty() );
        TS_ASSERT( buffer->size() == 0 );
        TS_ASSERT( buffer->capacity() == capacity );

        buffer->addData( boost::shared_ptr< std::string >( new std::string( "foo" ) ) );
        TS_ASSERT( !buffer->isEmpty() );
        TS_ASSERT( buffer->size() == 1 );
    }

    void test_addPop()
    {
        TS_TRACE( "test_addPop" );

        size_t capacity = 10;
        WLRingBuffer< size_t > buffer( capacity );
        TS_ASSERT( buffer.capacity() == capacity );

        boost::shared_ptr< size_t > element;

        for( size_t i = 0; i < 2.5 * capacity; ++i )
        {
            element.reset( newSizeT( i ) );
            buffer.addData( element );

            TS_ASSERT_EQUALS( *( buffer.popData().get() ), *( element.get() ) );
        }

        for( size_t i = 0; i < capacity / 2; ++i )
        {
            element.reset( newSizeT( i ) );
            buffer.addData( element );
        }

        for( size_t i = 0; i < capacity / 2; ++i )
        {
            TS_ASSERT_EQUALS( *( buffer.popData().get() ), i );
        }
    }

    void test_addGet()
    {
        TS_TRACE( "test_addPop" );

        size_t capacity = 10;
        WLRingBuffer< size_t > buffer( capacity );

        boost::shared_ptr< size_t > addElement;
        boost::shared_ptr< size_t > getElement;
        boost::shared_ptr< size_t > popElement;

        // Test for simple add, get pop
        addElement.reset( newSizeT( 4 ) );
        buffer.addData( addElement );
        getElement = buffer.getData();
        TS_ASSERT_EQUALS( *( addElement.get() ), *( getElement.get() ) );

        popElement = buffer.popData();
        TS_ASSERT_EQUALS( *( addElement.get() ), *( popElement.get() ) );

        getElement = buffer.getData( -1 );
        TS_ASSERT_EQUALS( *( addElement.get() ), *( getElement.get() ) );

        // Test for empty data exception
        try
        {
            getElement = buffer.getData( -2 );
            TS_FAIL( "There should be no data!" );
        }
        catch( const WException& e )
        {
            TS_ASSERT( true );
        }

        try
        {
            getElement = buffer.getData( 2 );
            TS_FAIL( "There should be no data!" );
        }
        catch( const WException& e )
        {
            TS_ASSERT( true );
        }

        const size_t addSize = capacity / 2;
        // Force a wrap around later
        for( size_t i = 0; i < addSize; ++i )
        {
            addElement.reset( newSizeT( i ) );
            buffer.addData( addElement );
            buffer.popData();
        }

        // Test for forward get
        for( size_t i = 0; i < addSize; ++i )
        {
            addElement.reset( newSizeT( i ) );
            buffer.addData( addElement );
        }

        for( size_t i = 0; i < addSize; ++i )
        {
            addElement.reset( newSizeT( i ) );
            getElement = buffer.getData( i );
            TS_ASSERT_EQUALS( *( addElement.get() ), *( getElement.get() ) );
        }

        // Test for backward get
        for( size_t i = 0; i < addSize; ++i )
        {
            addElement.reset( newSizeT( i ) );
            popElement = buffer.popData();
            TS_ASSERT_EQUALS( *( addElement.get() ), *( popElement.get() ) );
        }

        for( size_t i = 0; i < addSize; ++i )
        {
            addElement.reset( newSizeT( i ) );
            getElement = buffer.getData( -1 * addSize + i );
            TS_ASSERT_EQUALS( *( addElement.get() ), *( getElement.get() ) );
        }

        //TS_FAIL( "Not yet implemented!" );
    }

    void test_clear()
    {
        TS_TRACE( "test_clear" );

        size_t capacity = 10;
        WLRingBuffer< size_t > buffer( capacity );

        boost::shared_ptr< size_t > addElement;

        // Test for simple add, get pop
        addElement.reset( newSizeT( 4 ) );
        buffer.addData( addElement );
        buffer.addData( addElement );
        buffer.addData( addElement );
        TS_ASSERT_EQUALS( buffer.size(), 3 );
        buffer.clear();
        TS_ASSERT_EQUALS( buffer.size(), 0 );
        TS_ASSERT_EQUALS( buffer.capacity(), capacity );
    }

private:
    size_t* newSizeT( size_t value )
    {
        size_t* pointer = ( size_t* )malloc( sizeof(size_t) );
        *pointer = value;
        return pointer;
    }
};

#endif  // WLRINGBUFFER_TEST_H
