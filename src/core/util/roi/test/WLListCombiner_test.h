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

        boost::shared_ptr< WLListCombiner< int > > combiner( new WLListCombiner< int > );

        combiner->setFilter( a_ptr, b_ptr );
        a_ptr = combiner->getCombined();

        TS_ASSERT_EQUALS( a_ptr->size(), 6 );
    }
};

#endif  // WLLISTCOMBINER_TEST_H
