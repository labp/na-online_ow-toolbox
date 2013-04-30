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

#ifndef WFIRFILTER_TEST_H
#define WFIRFILTER_TEST_H

#include <algorithm>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <cxxtest/TestSuite.h>

#include <core/common/WLogger.h>

#include "core/data/WLDataSetEMM.h"
#include "core/data/emd/WLEMD.h"
#include "core/dataHandler/WDataSetEMMEEG.h"
#include "core/util/WLTimeProfiler.h"

#include "WFIRFilterTestHelper.h"

#include "../WFIRFilter.h"
#include "../WFIRFilterCpu.h"

#define EPS 0.0000001
#define ORDER 40
#define SFREQ 1000
#define C1FREQ 100
#define C2FREQ 1000

class WFIRFilterTest: public CxxTest::TestSuite
{
public:

    void setUp( void )
    {
        WLogger::startup();
    }

    void test_setGetCoefficientsVector( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );

        std::vector< double > coeffs;
        for( size_t i = 0; i < ORDER + 1; ++i )
        {
            coeffs.push_back( i );
        }
        filter.setCoefficients( coeffs );

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_setGetCoefficientsFile( void )
    {
        std::string fileName = W_FIXTURE_PATH + "22coeff.fcf";

        std::vector< double > coeffs;
        coeffs.push_back( -0.000284684171276985325052533148948441521 );
        coeffs.push_back( -0.000054211287529328173236814469859723431 );
        coeffs.push_back( 0.001562379810392586394762748369657856529 );
        coeffs.push_back( 0.006278424439878799515057927749239752302 );
        coeffs.push_back( 0.015978327462544458098525623768182413187 );
        coeffs.push_back( 0.032008748282184536215755343846467440017 );
        coeffs.push_back( 0.05430415922846235088350397290923865512 );
        coeffs.push_back( 0.080799670694357783595762612094404175878 );
        coeffs.push_back( 0.107502622756558613437150029312761034817 );
        coeffs.push_back( 0.129358490658292801800399729472701437771 );
        coeffs.push_back( 0.141691196060263757683372887186123989522 );
        coeffs.push_back( 0.141691196060263757683372887186123989522 );
        coeffs.push_back( 0.129358490658292801800399729472701437771 );
        coeffs.push_back( 0.107502622756558613437150029312761034817 );
        coeffs.push_back( 0.080799670694357783595762612094404175878 );
        coeffs.push_back( 0.05430415922846235088350397290923865512 );
        coeffs.push_back( 0.032008748282184536215755343846467440017 );
        coeffs.push_back( 0.015978327462544458098525623768182413187 );
        coeffs.push_back( 0.006278424439878799515057927749239752302 );
        coeffs.push_back( 0.001562379810392586394762748369657856529 );
        coeffs.push_back( -0.000054211287529328173236814469859723431 );
        coeffs.push_back( -0.000284684171276985325052533148948441521 );

        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );
        filter.setCoefficients( fileName.c_str() );

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_designLowpass( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::LOWPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );

        std::vector< double > coeffs;
        for( size_t i = 0; i < ORDER + 1; ++i )
        {
            // TODO (pieloth) set correct coefficients
            coeffs.push_back( i );
        }

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_designHighpass( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::HIGHPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );

        std::vector< double > coeffs;
        for( size_t i = 0; i < ORDER + 1; ++i )
        {
            // TODO (pieloth) set correct coefficients
            coeffs.push_back( i );
        }

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_designBandpass( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDPASS, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );

        std::vector< double > coeffs;
        for( size_t i = 0; i < ORDER + 1; ++i )
        {
            // TODO (pieloth) set correct coefficients
            coeffs.push_back( i );
        }

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_designBandstop( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDSTOP, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );

        std::vector< double > coeffs;
        for( size_t i = 0; i < ORDER + 1; ++i )
        {
            // TODO (pieloth) set correct coefficients
            coeffs.push_back( i );
        }

        std::vector< double > tmp = filter.getCoefficients();
        WFIRFilterTestHelper::isEqual( coeffs, tmp );
    }

    void test_previousDataSize( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDSTOP, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );
        const size_t channels = 42;
        const size_t samples = 666;

        // Test set/get previous data size //
        // Create test EMD
        LaBP::WDataSetEMMEEG::SPtr eeg( new LaBP::WDataSetEMMEEG() );
        boost::shared_ptr< LaBP::WLEMD::DataT > data( new LaBP::WLEMD::DataT() );
        data->resize( channels );
        for( size_t chan = 0; chan < channels; ++chan )
        {
            data->at( chan ).resize( samples );
        }
        eeg->setData( data );
        // Do size() test
        filter.storePreviousData( eeg );
        const LaBP::WLEMD::DataT& prevData = filter.getPreviousData( eeg );
        TS_ASSERT_EQUALS( channels, prevData.size() );
        TS_ASSERT_EQUALS( filter.m_coeffitients.size(), prevData.front().size() );
    }

    void test_previousDataContent( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDSTOP, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );
        const size_t channels = 42;
        const size_t samples = 666;

        // Test correct data content //
        // Create test EMD
        LaBP::WDataSetEMMEEG::SPtr eeg( new LaBP::WDataSetEMMEEG() );
        boost::shared_ptr< LaBP::WLEMD::DataT > data( new LaBP::WLEMD::DataT() );
        data->resize( channels );
        for( size_t c = 0; c < channels; ++c )
        {
            data->at( c ).reserve( samples );
            for( size_t s = 0; s < samples; ++s )
            {
                data->at( c ).push_back( c + s );
            }
        }
        eeg->setData( data );

        // Do test
        filter.storePreviousData( eeg );
        const LaBP::WLEMD::DataT& prevData = filter.getPreviousData( eeg );
        const size_t prevSamples = filter.m_coeffitients.size();
        size_t offset = samples - prevSamples;
        for( size_t c = 0; c < channels; ++c )
        {
            for( size_t s = 0; s < prevSamples; ++s )
            {
                TS_ASSERT_EQUALS( prevData[c][s], (*data)[c][s+offset] );
            }
        }
    }

    void test_doPostProcessing( void )
    {
        WFIRFilterCpu filter( WFIRFilter::WEFilterType::BANDSTOP, WFIRFilter::WEWindowsType::HAMMING, ORDER, SFREQ, C1FREQ,
                        C2FREQ );
        const size_t eChannels = 3;
        const size_t samples = 666;
        const size_t eSteps = 23;
        const size_t eMask = 1;
        const size_t shift = filter.m_coeffitients.size() / 2;

        // Test correct data content //
        LaBP::WLDataSetEMM::SPtr emmIn( new LaBP::WLDataSetEMM() );
        boost::shared_ptr< LaBP::WLDataSetEMM::EDataT > eventsIn( new LaBP::WLDataSetEMM::EDataT() );
        eventsIn->resize( eChannels );
        for( size_t c = 0; c < eChannels; ++c )
        {
            eventsIn->at( c ).resize( samples, 0 );
            for( size_t s = 0; s < samples; s += eSteps )
            {
                eventsIn->at( c )[s] = eMask;
            }
        }
        emmIn->setEventChannels( eventsIn );

        boost::shared_ptr< LaBP::WLDataSetEMM::EDataT > eventsPrev( new LaBP::WLDataSetEMM::EDataT() );
        eventsPrev->resize( eChannels );
        for( size_t c = 0; c < eChannels; ++c )
        {
            eventsPrev->at( c ).resize( samples, 0 );
        }
        LaBP::WLDataSetEMM::SPtr emmOut;

        // Do test //
        emmOut.reset( new LaBP::WLDataSetEMM() );

        filter.doPostProcessing( emmOut, emmIn, LaBP::WLTimeProfiler::SPtr() );
        boost::shared_ptr< LaBP::WLDataSetEMM::EDataT > eventsOut = emmOut->getEventChannels();
        for( size_t c = 0; c < eChannels; ++c )
        {
            size_t s = 0;
            for( ; s < shift; ++s )
            {
                TS_ASSERT_EQUALS( (*eventsOut)[c][s], ( *eventsPrev )[c][samples - shift + s] );
            }
            for( ; s < samples; ++s )
            {
                TS_ASSERT_EQUALS( ( *eventsOut )[c][s], ( *eventsIn )[c][s - shift] );
            }
        }

        eventsPrev = eventsIn;
        emmIn = emmOut;
        eventsIn = emmIn->getEventChannels();
        emmOut.reset( new LaBP::WLDataSetEMM() );

        filter.doPostProcessing( emmOut, emmIn, LaBP::WLTimeProfiler::SPtr() );
        eventsOut = emmOut->getEventChannels();
        for( size_t c = 0; c < eChannels; ++c )
        {
            size_t s = 0;
            for( ; s < shift; ++s )
            {
                TS_ASSERT_EQUALS( (*eventsOut)[c][s], ( *eventsPrev )[c][samples - shift + s] );
            }
            for( ; s < samples; ++s )
            {
                TS_ASSERT_EQUALS( ( *eventsOut )[c][s], ( *eventsIn )[c][s - shift] );
            }
        }
    }
};

#endif // WFIRFILTER_TEST_H
