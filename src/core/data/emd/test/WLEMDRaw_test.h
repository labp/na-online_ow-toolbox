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

#ifndef WLEMDRAWTEST_H_
#define WLEMDRAWTEST_H_

#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WOutOfBounds.h>

#include "../WLEMDRaw.h"

class WLEMDRaw_test: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_getDataPickError()
    {
        WLEMDRaw::SPtr raw = generateRaw( 42, 100 );
        WLEMDRaw::ChanPicksT picks1( raw->getNrChans() + 3 );
        TS_ASSERT_THROWS( raw->getData( picks1 ), WOutOfBounds );

        WLEMDRaw::ChanPicksT picks2( raw->getNrChans() - 3 );
        picks2.setOnes();
        picks2[1] = raw->getNrChans() + 3;
        TS_ASSERT_THROWS( raw->getData( picks2 ), WOutOfBounds );

        picks2.setOnes();
        picks2[1] = -1;
        TS_ASSERT_THROWS( raw->getData( picks2 ), WOutOfBounds );
    }

    void test_getDataPickAll()
    {
        const WLEMDRaw::ChanPicksT::Index nChannels = 42;
        const WLEMDRaw::ChanPicksT::Index nSamples = 100;

        WLEMDRaw::SPtr emdRaw = generateRaw( nChannels, nSamples );
        WLEMDRaw::ChanPicksT picks( nChannels );
        for( WLEMDRaw::ChanPicksT::Index i = 0; i < picks.size(); ++i )
        {
            picks[i] = i;
        }

        WLEMDRaw::DataSPtr dataPickedPtr = emdRaw->getData( picks );
        TS_ASSERT_EQUALS( picks.size(), dataPickedPtr->rows() );

        const size_t size = nChannels * nSamples;
        TS_ASSERT_SAME_DATA( emdRaw->getData().data(), dataPickedPtr->data(), size );

        // Check if it is a copy
        const WLEMDRaw::DataT& dataRaw = emdRaw->getData();
        WLEMDRaw::DataT& dataPicked = *dataPickedPtr;

        const WLEMData::ScalarT raw = dataRaw( picks[1], nSamples / 2 );
        dataPicked( 1, nSamples / 2 ) = raw * 2;
        TS_ASSERT_DIFFERS( dataPicked( 1, nSamples / 2 ), dataRaw( picks[1], nSamples / 2 ) );
        TS_ASSERT_EQUALS( dataRaw( picks[1], nSamples / 2 ), raw );
    }

    void test_getDataPickSparse()
    {
        const WLEMDRaw::ChanPicksT::Index nChannels = 42;
        const WLEMDRaw::ChanPicksT::Index nSamples = 100;

        WLEMDRaw::SPtr emdRaw = generateRaw( nChannels, nSamples );
        const WLEMDRaw::DataT& dataRaw = emdRaw->getData();
        WLEMDRaw::ChanPicksT picks( nChannels / 3 );
        for( WLEMDRaw::ChanPicksT::Index i = 0; i < picks.size(); ++i )
        {
            picks[i] = i + 2;
        }

        WLEMDRaw::DataSPtr dataPicked = emdRaw->getData( picks );
        TS_ASSERT_EQUALS( picks.size(), dataPicked->rows() );
        TS_ASSERT_DIFFERS( dataRaw.rows(), dataPicked->rows() );

        // Check if data is correct
        for( WLEMDRaw::ChanPicksT::Index i = 0; i < picks.size(); ++i )
        {
            const WLEMDRaw::ChannelT& chanRaw = dataRaw.row( picks[i] );
            const WLEMDRaw::ChannelT& chanPicked = dataPicked->row( i );
            for( WLEMDRaw::ChanPicksT::Index s = 0; s < nSamples; ++s )
            {
                TS_ASSERT_EQUALS( chanRaw( s ), chanPicked( s ) );
            }
        }
    }

private:
    static WLEMDRaw::SPtr generateRaw( size_t nChannels, size_t nSamples )
    {
        WLEMDRaw::SPtr raw( new WLEMDRaw() );
        WLEMDRaw::DataSPtr data( new WLEMDRaw::DataT( WLEMDRaw::DataT::Random( nChannels, nSamples ) ) );
        raw->setData( data );
        return raw;
    }

};

#endif  // WLEMDRAWTEST_H_
