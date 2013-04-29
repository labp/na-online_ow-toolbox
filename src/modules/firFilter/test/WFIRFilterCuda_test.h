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

        filter.reset(
                        new WFIRFilterCuda( WFIRFilter::WEFilterType::BANDPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ,
                                        C1FREQ, C2FREQ ) );
        WFIRFilterTestHelper::filterImpulseTest( filter );
    }

    void test_filterStep( void )
    {
        std::string fileName = W_FIXTURE_PATH + COEFFILE;
        WFIRFilterCuda::SPtr filter( new WFIRFilterCuda( fileName.c_str() ) );
        WFIRFilterTestHelper::filterStepTest( filter );

        filter.reset(
                        new WFIRFilterCuda( WFIRFilter::WEFilterType::BANDPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ,
                                        C1FREQ, C2FREQ ) );
        WFIRFilterTestHelper::filterStepTest( filter );
    }
};

#endif // WFIRFILTERCUDA_TEST_H
