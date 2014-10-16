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

#ifndef WFIRFILTERTESTHELPER_H_
#define WFIRFILTERTESTHELPER_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
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
    static void isEqual( const WLEMData::ChannelT& vec1, const WLEMData::ChannelT& vec2, double d = EPS )
    {
        TS_ASSERT_EQUALS( vec1.size(), vec2.size() );

        size_t count = std::min( vec1.size(), vec2.size() );
        for( size_t i = 0; i < count; ++i )
        {
            TS_ASSERT_DELTA( vec1( i ), vec2( i ), d );
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
        for( WLEMData::ChannelT::Index i = coefficients; i < outExpected.size(); ++i )
        {
            outExpected( i ) = firSum;
        }

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected, out.row( chan ) );
        }
    }

    /**
     * See http://www.dspguru.com/dsp/faqs/fir/implementation
     */
    static void filterSineTest( WFIRFilter::SPtr filter )
    {
        filter->reset();
        filterSineLowpassTest( filter );
        filterSineHighpassTest( filter );
        filterSineBandpassTest( filter );
        filterSineBandstopTest( filter );
    }

private:
    static void generateSinusWave( WLEMData::DataT* const in, float sr, float f, float amp, float offset = 0 )
    {
        float delta = 1 / static_cast< float >( sr );
        for( WLEMData::DataT::Index row = 0; row < in->rows(); ++row )
        {
            for( WLEMData::ChannelT::Index col = 0; col < in->cols(); ++col )
            {
                const WLEMData::ScalarT x = col * delta;
                const WLEMData::ScalarT x_rad = 2 * M_PI * x;
                const WLEMData::ScalarT y = amp * sin( f * x_rad ) + offset;
                ( *in )( row, col ) = y;
            }
        }
    }

    static void filterSineLowpassTest( WFIRFilter::SPtr filter )
    {
        // Test with *.fcf file from Matlab fdatool //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoLowpassTest() with fcf.";
        const std::string fName = W_FIXTURE_PATH + "lp_hamming_o200fs1000fc50.fcf";
        filter->reset();
        filter->setCoefficients( fName );
        filterSineDoLowpassTest( filter, 200, 1000.0 );

        // Test with designed coefficients //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoLowpassTest() with designed coefficients.";
        // Setup lowpass filter
        const WFIRFilter::WEFilterType::Enum type = WFIRFilter::WEFilterType::LOWPASS;
        const WFIRFilter::WEWindowsType::Enum windows = WFIRFilter::WEWindowsType::HAMMING;
        const size_t order = 200;
        const float f_cutoff = 50.0;
        const WFIRFilter::ScalarT sampling_frequency = 1000;
        filter->reset();
        filter->setCutOffFrequency1( f_cutoff, false );
        filter->setFilterType( type, false );
        filter->setOrder( order, false );
        filter->setSamplingFrequency( sampling_frequency, false );
        filter->setWindowsType( windows, false );
        filter->design();

         filterSineDoLowpassTest( filter, 200, 1000.0 );
    }

    static void filterSineDoLowpassTest( WFIRFilter::SPtr filter, const size_t order, const float fs )
    {
        // Parameter
        // ---------
        // first #order samples shift and tune in, last #order samples shift into next block
        const size_t samples = order * 4;
        const size_t channels = 1;
        const float offset = 5;
        const float amp = 10; // amplitude factor
        // attenuation at 100Hz: -65dB (Matlab fdatool)
        const float att_factor_np = pow( 10, -65.0 / 20.0 ); // attenuation factor, no pass
        const float diff = 0.1; // attenuation factor, pass
        float f; // frequency
        WLEMData::DataSPtr in;

        // Setup data with 100Hz - no pass!
        // --------------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 100.0;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn1( new WLEMDEEG() );
        emdIn1->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut1 = filter->filter( emdIn1 );
        const WLEMData::DataT& out1 = emdOut1->getData();
        // just check the mid part samples, due to shift and tune in
        WLEMData::ChannelT outExpected1( 2 * order );
        outExpected1.setConstant( offset );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected1, out1.row( chan ).block( 0, order, 1, 2 * order ), amp * att_factor_np );
        }

        // Setup data with 5Hz - pass!
        // ---------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 5;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn2( new WLEMDEEG() );
        emdIn2->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut2 = filter->filter( emdIn2 );
        const WLEMData::DataT& out2 = emdOut2->getData();
        WLEMData::ChannelT outExpected2( 2 * order );
        outExpected2 = in->block( 0, order / 2, 1, 2 * order );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected2, out2.row( chan ).block( 0, order, 1, 2 * order ), diff );
        }
    }

    static void filterSineHighpassTest( WFIRFilter::SPtr filter )
    {
        // Test with *.fcf file from Matlab fdatool //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoHighpassTest() with fcf.";
        const std::string fName = W_FIXTURE_PATH + "hp_hamming_o200fs1000fc50.fcf";
        filter->reset();
        filter->setCoefficients( fName );
        filterSineDoHighpassTest( filter, 200, 1000.0 );

        // Test with designed coefficients //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoHighpassTest() with designed coefficients.";
        // Setup lowpass filter
        const WFIRFilter::WEFilterType::Enum type = WFIRFilter::WEFilterType::HIGHPASS;
        const WFIRFilter::WEWindowsType::Enum windows = WFIRFilter::WEWindowsType::HAMMING;
        const size_t order = 200;
        const float f_cutoff = 50.0;
        const WFIRFilter::ScalarT sampling_frequency = 1000;
        filter->reset();
        filter->setCutOffFrequency1( f_cutoff, false );
        filter->setFilterType( type, false );
        filter->setOrder( order, false );
        filter->setSamplingFrequency( sampling_frequency, false );
        filter->setWindowsType( windows, false );
        filter->design();

         filterSineDoHighpassTest( filter, 200, 1000.0 );
    }

    static void filterSineDoHighpassTest( WFIRFilter::SPtr filter, const size_t order, const float fs )
    {
        // Parameter
        // ---------
        // first #order samples shift and tune in, last #order samples shift into next block
        const size_t samples = order * 4;
        const size_t channels = 1;
        const float offset = 0;
        const float amp = 10; // amplitude factor
        // attenuation at 25Hz: -55dB (Matlab fdatool)
        const float att_factor_np = pow( 10, -50.0 / 20.0 ); // attenuation factor, no pass
        const float diff = 0.1; // attenuation factor, pass
        float f; // frequency
        WLEMData::DataSPtr in;

        // Setup data with 25Hz - no pass!
        // --------------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 25.0;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn1( new WLEMDEEG() );
        emdIn1->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut1 = filter->filter( emdIn1 );
        const WLEMData::DataT& out1 = emdOut1->getData();
        // just check the mid part samples, due to shift and tune in
        WLEMData::ChannelT outExpected1( 2 * order );
        outExpected1.setConstant( 0 );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected1, out1.row( chan ).block( 0, order, 1, 2 * order ), amp * att_factor_np );
        }

        // Setup data with 100Hz - pass!
        // ---------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 100;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn2( new WLEMDEEG() );
        emdIn2->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut2 = filter->filter( emdIn2 );
        const WLEMData::DataT& out2 = emdOut2->getData();
        WLEMData::ChannelT outExpected2( 2 * order );
        outExpected2 = in->block( 0, order / 2, 1, 2 * order );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected2, out2.row( chan ).block( 0, order, 1, 2 * order ), diff );
        }
    }

    static void filterSineBandpassTest( WFIRFilter::SPtr filter )
    {
        // Test with *.fcf file from Matlab fdatool //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoBandpassTest() with fcf.";
        const std::string fName = W_FIXTURE_PATH + "bp_hamming_o200fs1000fc50_100.fcf";
        filter->reset();
        filter->setCoefficients( fName );
        filterSineDoBandpassTest( filter, 200, 1000.0 );

        // Test with designed coefficients //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoBandpassTest() with designed coefficients.";
        // Setup lowpass filter
        const WFIRFilter::WEFilterType::Enum type = WFIRFilter::WEFilterType::BANDPASS;
        const WFIRFilter::WEWindowsType::Enum windows = WFIRFilter::WEWindowsType::HAMMING;
        const size_t order = 200;
        const float fc1 = 50.0;
        const float fc2 = 100.0;
        const WFIRFilter::ScalarT sampling_frequency = 1000;
        filter->reset();
        filter->setCutOffFrequency1( fc1, false );
        filter->setCutOffFrequency2( fc2, false );
        filter->setFilterType( type, false );
        filter->setOrder( order, false );
        filter->setSamplingFrequency( sampling_frequency, false );
        filter->setWindowsType( windows, false );
        filter->design();

         filterSineDoBandpassTest( filter, 200, 1000.0 );
    }

    static void filterSineDoBandpassTest( WFIRFilter::SPtr filter, const size_t order, const float fs )
    {
        // Parameter
        // ---------
        // first #order samples shift and tune in, last #order samples shift into next block
        const size_t samples = order * 4;
        const size_t channels = 1;
        const float offset = 0;
        const float amp = 10; // amplitude factor
        // attenuation at 25Hz/125Hz: -60dB (Matlab fdatool)
        const float att_factor_np = pow( 10, -60.0 / 20.0 ); // attenuation factor, no pass
        const float diff = 0.1; // attenuation factor, pass
        float f; // frequency
        WLEMData::DataSPtr in;

        // Setup data with 25Hz - no pass!
        // --------------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 25.0;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn1( new WLEMDEEG() );
        emdIn1->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut1 = filter->filter( emdIn1 );
        const WLEMData::DataT& out1 = emdOut1->getData();
        // just check the mid part samples, due to shift and tune in
        WLEMData::ChannelT outExpected1( 2 * order );
        outExpected1.setConstant( 0 );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected1, out1.row( chan ).block( 0, order, 1, 2 * order ), amp * att_factor_np );
        }

        // Setup data with 125Hz - no pass!
        // --------------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 125.0;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn2( new WLEMDEEG() );
        emdIn2->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut2 = filter->filter( emdIn2 );
        const WLEMData::DataT& out2 = emdOut2->getData();
        // just check the mid part samples, due to shift and tune in
        WLEMData::ChannelT outExpected2( 2 * order );
        outExpected2.setConstant( 0 );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected2, out2.row( chan ).block( 0, order, 1, 2 * order ), amp * att_factor_np );
        }

        // Setup data with 75Hz - pass!
        // ---------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 75;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn3( new WLEMDEEG() );
        emdIn3->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut3 = filter->filter( emdIn3 );
        const WLEMData::DataT& out3 = emdOut3->getData();
        WLEMData::ChannelT outExpected3( 2 * order );
        outExpected3 = in->block( 0, order / 2, 1, 2 * order );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected3, out3.row( chan ).block( 0, order, 1, 2 * order ), diff );
        }
    }

    static void filterSineBandstopTest( WFIRFilter::SPtr filter )
    {
        // Test with *.fcf file from Matlab fdatool //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoBandstopTest() with fcf.";
        const std::string fName = W_FIXTURE_PATH + "bs_hamming_o200fs1000fc50_100.fcf";
        filter->reset();
        filter->setCoefficients( fName );
        filterSineDoBandstopTest( filter, 200, 1000.0 );

        // Test with designed coefficients //
        wlog::info( "WFIRFilterTestHelper" ) << "filterSineDoBandstopTest() with designed coefficients.";
        // Setup lowpass filter
        const WFIRFilter::WEFilterType::Enum type = WFIRFilter::WEFilterType::BANDSTOP;
        const WFIRFilter::WEWindowsType::Enum windows = WFIRFilter::WEWindowsType::HAMMING;
        const size_t order = 200;
        const float fc1 = 50.0;
        const float fc2 = 100.0;
        const WFIRFilter::ScalarT sampling_frequency = 1000;
        filter->reset();
        filter->setCutOffFrequency1( fc1, false );
        filter->setCutOffFrequency2( fc2, false );
        filter->setFilterType( type, false );
        filter->setOrder( order, false );
        filter->setSamplingFrequency( sampling_frequency, false );
        filter->setWindowsType( windows, false );
        filter->design();

         filterSineDoBandstopTest( filter, 200, 1000.0 );
    }

    static void filterSineDoBandstopTest( WFIRFilter::SPtr filter, const size_t order, const float fs )
    {
        // Parameter
        // ---------
        // first #order samples shift and tune in, last #order samples shift into next block
        const size_t samples = order * 4;
        const size_t channels = 1;
        const float offset = 5;
        const float amp = 10; // amplitude factor
        // attenuation at 25Hz/125Hz: -60dB (Matlab fdatool)
        const float att_factor_np = pow( 10, -50.0 / 20.0 ); // attenuation factor, no pass
        const float diff = 0.1; // attenuation factor, pass
        float f; // frequency
        WLEMData::DataSPtr in;

        // Setup data with 75Hz - no pass!
        // --------------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 75.0;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn1( new WLEMDEEG() );
        emdIn1->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut1 = filter->filter( emdIn1 );
        const WLEMData::DataT& out1 = emdOut1->getData();
        // just check the mid part samples, due to shift and tune in
        WLEMData::ChannelT outExpected1( 2 * order );
        outExpected1.setConstant( 5 );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected1, out1.row( chan ).block( 0, order, 1, 2 * order ), amp * att_factor_np );
        }

        // Setup data with 25Hz - pass!
        // ---------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 25;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn2( new WLEMDEEG() );
        emdIn2->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut2 = filter->filter( emdIn2 );
        const WLEMData::DataT& out2 = emdOut2->getData();
        WLEMData::ChannelT outExpected2( 2 * order );
        outExpected2 = in->block( 0, order / 2, 1, 2 * order );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected2, out2.row( chan ).block( 0, order, 1, 2 * order ), diff );
        }

        // Setup data with 125Hz - pass!
        // ---------------------------
        in.reset( new WLEMData::DataT( channels, samples ) );
        f = 125;
        generateSinusWave( in.get(), fs, f, amp, offset );
        WLEMData::SPtr emdIn3( new WLEMDEEG() );
        emdIn3->setData( in );

        filter->reset();
        WLEMData::SPtr emdOut3 = filter->filter( emdIn3 );
        const WLEMData::DataT& out3 = emdOut3->getData();
        WLEMData::ChannelT outExpected3( 2 * order );
        outExpected3 = in->block( 0, order / 2, 1, 2 * order );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            isEqual( outExpected3, out3.row( chan ).block( 0, order, 1, 2 * order ), diff );
        }
    }
};

#endif  // WFIRFILTERTESTHELPER_H_
