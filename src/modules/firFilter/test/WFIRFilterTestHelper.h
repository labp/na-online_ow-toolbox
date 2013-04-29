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

#ifndef WFIRFILTERTESTHELPER_H_
#define WFIRFILTERTESTHELPER_H_

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cxxtest/TestSuite.h>

#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/dataHandler/WDataSetEMMEEG.h"

#include "core/util/WLTimeProfiler.h"

#include "../WFIRFilter.h"

#define EPS 0.0000001
#define CHANNELS 42

class WFIRFilterTestHelper
{
public:
    static void isEqual( std::vector< double >& vec1, std::vector< double >& vec2 )
    {
        TS_ASSERT_EQUALS( vec1.size(), vec2.size() );

        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DELTA( vec1[i], vec2[i], EPS );
        }
    }

    static void isNotEqual( std::vector< double >& vec1, std::vector< double >& vec2 )
    {
        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DIFFERS( vec1[i], vec2[i] );
        }
    }

    /**
     * See from http://www.dspguru.com/dsp/faqs/fir/implementation
     */
    static void filterImpulseTest( WFIRFilter::SPtr filter )
    {
        const size_t coefficients = filter->getCoefficients().size();
        const size_t channels = CHANNELS;
        const size_t samples = coefficients;

        boost::shared_ptr< std::vector< std::vector< double > > > in( new std::vector< std::vector< double > >() );
        for( size_t chan = 0; chan < channels; ++chan )
        {
            std::vector< double > channel;
            channel.push_back( 1 );
            for( size_t samps = 1; samps < samples; ++samps )
            {
                channel.push_back( 0 );
            }
            in->push_back( channel );
        }
        LaBP::WDataSetEMMEMD::SPtr emdIn( new LaBP::WDataSetEMMEEG() );
        emdIn->setData( in );

        LaBP::WDataSetEMMEMD::SPtr emdOut = filter->filter( emdIn, LaBP::WLTimeProfiler::SPtr() );

        std::vector< double > outExpected = filter->getCoefficients();
        std::vector< std::vector< double > > out = emdOut->getData();

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected, out[chan] );
        }
    }

    /**
     * See http://www.dspguru.com/dsp/faqs/fir/implementation
     */
    static void filterStepTest( WFIRFilter::SPtr filter )
    {
        const size_t coefficients = filter->getCoefficients().size();
        const size_t channels = CHANNELS;
        const size_t samples = coefficients * 2;

        boost::shared_ptr< std::vector< std::vector< double > > > in( new std::vector< std::vector< double > >() );
        for( size_t chan = 0; chan < channels; ++chan )
        {
            std::vector< double > channel;
            for( size_t samps = 0; samps < samples; ++samps )
            {
                channel.push_back( 1 );
            }
            in->push_back( channel );
        }
        LaBP::WDataSetEMMEMD::SPtr emdIn( new LaBP::WDataSetEMMEEG() );
        emdIn->setData( in );

        LaBP::WDataSetEMMEMD::SPtr emdOut = filter->filter( emdIn, LaBP::WLTimeProfiler::SPtr() );

        double firSum = 0;
        for( size_t i = 0; i < coefficients; ++i )
        {
            firSum += filter->getCoefficients()[i];
        }
        std::vector< std::vector< double > > out = emdOut->getData();
        std::vector< double > outExpected = out.front();
        for( size_t i = coefficients; i < outExpected.size(); ++i )
        {
            outExpected[i] = firSum;
        }

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected, out[chan] );
        }
    }
};

#endif /* WFIRFILTERTESTHELPER_H_ */
