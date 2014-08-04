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

#ifndef WLLISTCOMBINER_TEST_H
#define WLLISTCOMBINER_TEST_H

#include <list>

#include <cxxtest/TestSuite.h>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "core/util/roi/filterCombiner/WLListCombiner.h"

class WLListCombinerTest: public CxxTest::TestSuite
{
public:

    void setUp( void )
    {
        WLogger::startup();
    }

    /**
     * Unit test to test the filter combiner for a WLListCombiner.
     */
    void test_combineList()
    {
        TS_TRACE( "combineList" );

        boost::shared_ptr< std::list< int > > a_ptr( new std::list< int > );
        boost::shared_ptr< std::list< int > > b_ptr( new std::list< int > );

        a_ptr->push_back( 1 );
        a_ptr->push_back( 3 );
        a_ptr->push_back( 5 );

        b_ptr->push_back( 2 );
        b_ptr->push_back( 4 );
        b_ptr->push_back( 5 );
        b_ptr->push_back( 6 );

        WLListCombiner< int >::SPtr combiner( new WLListCombiner< int > );

        combiner->setFilter< std::list< int > >( a_ptr, b_ptr );
        a_ptr = combiner->getFilter< std::list< int > >();

        TS_ASSERT_EQUALS( a_ptr->size(), 6 );
    }
};

#endif // WLLISTCOMBINER_TEST_H
