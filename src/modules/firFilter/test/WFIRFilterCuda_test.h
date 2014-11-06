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

#ifndef WFIRFILTERCUDA_TEST_H
#define WFIRFILTERCUDA_TEST_H

#include <string>

#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>

#include "../WFIRFilter.h"
#include "../WFIRFilterCuda.h"

#include "WFIRFilterTestHelper.h"

#define COEFFILE "22coeff.fcf"
#define ORDER 40
#define SFREQ 1000
#define C1FREQ 100
#define C2FREQ 1000

class WFIRFilterCudaTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        // WFIRFilter* class needs a WLogger.
        WLogger::startup();
    }

    void test_filterImpulse( void )
    {
        std::string fileName = W_FIXTURE_PATH + COEFFILE;
        WFIRFilterCuda::SPtr filter( new WFIRFilterCuda( fileName.c_str() ) );
        WFIRFilterTestHelper::filterImpulseTest( filter );

        filter.reset( new WFIRFilterCuda( WFIRFilter::WEFilterType::BANDPASS, WLWindowsFunction::HAMMING, ORDER, SFREQ,
        C1FREQ, C2FREQ ) );
        WFIRFilterTestHelper::filterImpulseTest( filter );
    }

    void test_filterStep( void )
    {
        std::string fileName = W_FIXTURE_PATH + COEFFILE;
        WFIRFilterCuda::SPtr filter( new WFIRFilterCuda( fileName.c_str() ) );
        WFIRFilterTestHelper::filterStepTest( filter );

        filter.reset( new WFIRFilterCuda( WFIRFilter::WEFilterType::BANDPASS, WLWindowsFunction::HAMMING, ORDER, SFREQ,
        C1FREQ, C2FREQ ) );
        WFIRFilterTestHelper::filterStepTest( filter );
    }

    void test_filterSine( void )
    {
        std::string fileName = W_FIXTURE_PATH + COEFFILE;
        WFIRFilterCuda::SPtr filter( new WFIRFilterCuda( fileName.c_str() ) );
        WFIRFilterTestHelper::filterSineTest( filter );

        filter.reset( new WFIRFilterCuda( WFIRFilter::WEFilterType::BANDPASS, WLWindowsFunction::HAMMING, ORDER, SFREQ,
        C1FREQ, C2FREQ ) );
        WFIRFilterTestHelper::filterSineTest( filter );
    }
};

#endif  // WFIRFILTERCUDA_TEST_H
