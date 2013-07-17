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

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "../WFIRFilter.h"

#define EPS 0.0000001
#define CHANNELS 42

class WFIRFilterTestHelper
{
public:
    static void isEqual( const WLEMData::ChannelT& vec1, const WLEMData::ChannelT& vec2 )
    {
        TS_ASSERT_EQUALS( vec1.size(), vec2.size() );

        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DELTA( vec1( i ), vec2( i ), EPS );
        }
    }

    static void isNotEqual( const WLEMData::ChannelT& vec1, const WLEMData::ChannelT& vec2 )
    {
        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DIFFERS( vec1( i ), vec2( i ) );
        }
    }

    static void isEqual( const std::vector< WFIRFilter::ScalarT >& vec1, const std::vector< WFIRFilter::ScalarT >& vec2 )
    {
        TS_ASSERT_EQUALS( vec1.size(), vec2.size() );

        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DELTA( vec1[i], vec2[i], EPS );
        }
    }

    static void isNotEqual( const std::vector< WFIRFilter::ScalarT >& vec1, const std::vector< WFIRFilter::ScalarT >& vec2 )
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

        WLEMData::DataSPtr in( new WLEMData::DataT( channels, samples ) );
        in->setZero();

        for( size_t chan = 0; chan < channels; ++chan )
        {
            ( *in )( chan, 0 ) = 1;
        }
        WLEMData::SPtr emdIn( new WLEMDEEG() );
        emdIn->setData( in );

        WLEMData::SPtr emdOut = filter->filter( emdIn );

        WLEMData::ChannelT outExpected = WLEMData::ChannelT::Map( filter->getCoefficients().data(), coefficients );
        const WLEMData::DataT& out = emdOut->getData();

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected, out.row( chan ) );
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

        WLEMData::DataSPtr in( new WLEMData::DataT( channels, samples ) );
        in->setOnes();
        WLEMData::SPtr emdIn( new WLEMDEEG() );
        emdIn->setData( in );

        WLEMData::SPtr emdOut = filter->filter( emdIn );

        WLEMData::ScalarT firSum = 0;
        for( size_t i = 0; i < coefficients; ++i )
        {
            firSum += filter->getCoefficients()[i];
        }

        const WLEMData::DataT& out = emdOut->getData();
        WLEMData::ChannelT outExpected = out.row( 0 );
        for( size_t i = coefficients; i < outExpected.size(); ++i )
        {
            outExpected( i ) = firSum;
        }

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected, out.row( chan ) );
        }
    }
};

#endif /* WFIRFILTERTESTHELPER_H_ */
