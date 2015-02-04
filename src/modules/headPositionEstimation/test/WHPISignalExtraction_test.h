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

#ifndef WHEADPOSITIONESTIMATION_TEST_H_
#define WHEADPOSITIONESTIMATION_TEST_H_

#include <cxxtest/TestSuite.h>
#include <core/common/WLogger.h>

#include "../WHPISignalExtraction.h"

#define DELTA 1e-6

/**
 * \author pieloth
 */
class WHPISignalExtractionTest: public CxxTest::TestSuite
{
public:

    void setUp()
    {
        WLogger::startup( std::cout, LL_DEBUG );
    }

    void test_getSetWindowsSize()
    {
        WHPISignalExtraction hpe;
        const WLTimeT expected = 0.300 * WLUnits::s;
        hpe.setWindowsSize( expected );
        const WLTimeT actual = hpe.getWindowsSize();
        TS_ASSERT_DELTA( expected.value(), actual.value(), DELTA );
    }

    void test_getSetStepSize()
    {
        WHPISignalExtraction hpe;
        const WLTimeT expected = 42.0 * WLUnits::s;
        hpe.setStepSize( expected );
        const WLTimeT actual = hpe.getStepSize();
        TS_ASSERT_DELTA( expected.value(), actual.value(), DELTA );
    }

    void test_addGetClearFrequencies()
    {
        WHPISignalExtraction hpe;
        const WLFreqT hpi1 = 154.0 * WLUnits::Hz;
        hpe.addFrequency( hpi1 );
        const WLFreqT hpi2 = 158.0 * WLUnits::Hz;
        hpe.addFrequency( hpi2 );
        const WLFreqT hpi3 = 162.0 * WLUnits::Hz;
        hpe.addFrequency( hpi3 );
        const WLFreqT hpi4 = 166.0 * WLUnits::Hz;
        hpe.addFrequency( hpi4 );
        const WLFreqT hpi5 = 170.0 * WLUnits::Hz;
        hpe.addFrequency( hpi5 );

        std::vector< WLFreqT > actualFreqs = hpe.getFrequencies();
        TS_ASSERT_EQUALS( actualFreqs.size(), 5 );

        TS_ASSERT_DELTA( actualFreqs[0].value(), hpi1.value(), DELTA );
        TS_ASSERT_DELTA( actualFreqs[1].value(), hpi2.value(), DELTA );
        TS_ASSERT_DELTA( actualFreqs[2].value(), hpi3.value(), DELTA );
        TS_ASSERT_DELTA( actualFreqs[3].value(), hpi4.value(), DELTA );
        TS_ASSERT_DELTA( actualFreqs[4].value(), hpi5.value(), DELTA );

        hpe.clearFrequencies();
        actualFreqs = hpe.getFrequencies();
        TS_ASSERT( actualFreqs.empty() );
    }

    void test_prepare()
    {
        WHPISignalExtraction hpe;
        const WLFreqT hpi1 = 154.0 * WLUnits::Hz;
        hpe.addFrequency( hpi1 );
        const WLFreqT hpi2 = 158.0 * WLUnits::Hz;
        hpe.addFrequency( hpi2 );
        const WLFreqT hpi3 = 162.0 * WLUnits::Hz;
        hpe.addFrequency( hpi3 );
        const WLFreqT hpi4 = 166.0 * WLUnits::Hz;
        hpe.addFrequency( hpi4 );
        const WLFreqT hpi5 = 170.0 * WLUnits::Hz;
        hpe.addFrequency( hpi5 );

        hpe.setSamplingFrequency( 500.0 * WLUnits::Hz );

        TS_ASSERT( hpe.prepare() );
    }

    void test_estimateAmplitueds()
    {
        WHPISignalExtraction hpe;
        const WLFreqT hpi1 = 154.0 * WLUnits::Hz;
        hpe.addFrequency( hpi1 );
        const WLFreqT hpi2 = 158.0 * WLUnits::Hz;
        hpe.addFrequency( hpi2 );
        const WLFreqT hpi3 = 162.0 * WLUnits::Hz;
        hpe.addFrequency( hpi3 );
        const WLFreqT hpi4 = 166.0 * WLUnits::Hz;
        hpe.addFrequency( hpi4 );
        const WLFreqT hpi5 = 170.0 * WLUnits::Hz;
        hpe.addFrequency( hpi5 );
        hpe.setStepSize( 0.010 * WLUnits::s );
        hpe.setWindowsSize( 0.200 * WLUnits::s );
        hpe.setSamplingFrequency( 1000.0 * WLUnits::Hz );
        TS_ASSERT( hpe.prepare() );

        // TODO (pieloth): Implement test case.
        WLEMDMEG::SPtr meg( new WLEMDMEG );
        WLEMDMEG::DataSPtr data( new WLEMDMEG::DataT( WLEMDMEG::DataT::Random( 306, 1000 ) ) );
        meg->setData( data );
        WLEMDHPI::SPtr hpiOut( new WLEMDHPI );
        TS_FAIL( "Test not implemented!" );
//        TS_ASSERT( hpe.reconstructAmplitudes( hpiOut, meg ) );
    }

};

#endif  // WHEADPOSITIONESTIMATION_TEST_H_
