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

#include <cmath>
#include <algorithm> // std::min

#include <core/common/WLogger.h>

#include "core/preprocessing/WLPreprocessing.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WHPISignalExtraction.h"
#include "../../core/preprocessing/WLWindowFunction.h"

const std::string WHPISignalExtraction::CLASS = "WHPISignalExtraction";

static const WLTimeT WINDOWS_SIZE = 0.2 * WLUnits::s;
static const WLTimeT STEP_SIZE = 0.01 * WLUnits::s;
static const WLFreqT SAMPLING_FREQ = 1000.0 * WLUnits::Hz;
static const double MIN_WINDOWS_PERIODS = 20.0;

WHPISignalExtraction::WHPISignalExtraction() :
                m_isPrepared( false ), m_windowsSize( WINDOWS_SIZE ), m_stepSize( STEP_SIZE ), m_sampFreq( SAMPLING_FREQ )
{
    m_angFrequencies.reserve( 5 );
}

WHPISignalExtraction::~WHPISignalExtraction()
{
}

WLTimeT WHPISignalExtraction::getWindowsSize() const
{
    return m_windowsSize;
}

WLTimeT WHPISignalExtraction::setWindowsSize( WLTimeT winSize )
{
    if( !m_angFrequencies.empty() )
    {
        const std::vector< WLFreqT >& freqs = getFrequencies();
        std::vector< WLFreqT >::const_iterator it = freqs.begin();
        WLFreqT fmin = *it;
        for( it = it + 1; it != freqs.end(); ++it )
        {
            fmin = std::min( fmin, *it );
        }

        const float N = fmin * winSize;
        if( N < MIN_WINDOWS_PERIODS )
        {
            const WLTimeT wsize_old = winSize;
            winSize = MIN_WINDOWS_PERIODS / fmin;
            wlog::warn( CLASS ) << "Windows size is to small: t_old=" << wsize_old << " t_new=" << winSize;
        }
    }
    m_windowsSize = winSize;
    m_isPrepared = false;
    return m_windowsSize;
}

WLTimeT WHPISignalExtraction::getStepSize() const
{
    return m_stepSize;
}

void WHPISignalExtraction::setStepSize( WLTimeT stepSize )
{
    m_stepSize = stepSize;
    m_isPrepared = false;
}

void WHPISignalExtraction::addFrequency( WLFreqT freq )
{
    const WLFreqT angFreq = 2 * M_PI * freq;
    m_angFrequencies.push_back( angFreq );
    wlog::debug( CLASS ) << "Added angular frequency: " << angFreq;
    m_isPrepared = false;
}

std::vector< WLFreqT > WHPISignalExtraction::getFrequencies() const
{
    std::vector< WLFreqT > frequencies;
    frequencies.reserve( m_angFrequencies.size() );
    std::vector< WLFreqT >::const_iterator it;
    for( it = m_angFrequencies.begin(); it != m_angFrequencies.end(); ++it )
    {
        const WLFreqT freq = *it / ( 2 * M_PI );
        frequencies.push_back( freq );
    }
    return frequencies;
}

size_t WHPISignalExtraction::clearFrequencies()
{
    const size_t removed = m_angFrequencies.size();
    m_angFrequencies.clear();
    m_isPrepared = false;
    return removed;
}

WLFreqT WHPISignalExtraction::getSamplingFrequency() const
{
    return m_sampFreq;
}

void WHPISignalExtraction::setSamplingFrequency( WLFreqT sfreq )
{
    m_sampFreq = sfreq;
    m_isPrepared = false;
}

bool WHPISignalExtraction::prepare()
{
    wlog::debug( CLASS ) << "prepare() called!";
    WLTimeProfiler prof( CLASS, __func__ );

    if( m_isPrepared )
    {
        return m_isPrepared;
    }
    if( m_sampFreq < 0.0 * WLUnits::Hz )
    {
        wlog::error( CLASS ) << "Wrong sampling frequency.";
        return m_isPrepared;
    }
    if( m_angFrequencies.size() < 3 )
    {
        wlog::error( CLASS ) << "At least 3 HPI coils are needed!";
        return m_isPrepared;
    }

    const MatrixT::Index J = m_angFrequencies.size();
    const MatrixT::Index N = m_windowsSize * m_sampFreq;
    wlog::debug( CLASS ) << "J=" << J;
    wlog::debug( CLASS ) << "N=" << N;
    MatrixT m_a( N, 2 * J );

    const WLTimeT T = 1.0 / m_sampFreq;
    wlog::debug( CLASS ) << "T=" << T;

    for( MatrixT::Index i = 0; i < N; ++i )
    {
        for( MatrixT::Index j = 0; j < J; ++j )
        {
            const double tmp = m_angFrequencies[j] * T;
            m_a( i, j ) = sin( tmp * i );
            m_a( i, j + J ) = cos( tmp * i );
        }
    }
    wlog::debug( CLASS ) << "A=" << m_a.rows() << "x" << m_a.cols();

    m_at = m_a.transpose();
    m_ata = m_at * m_a;
    wlog::debug( CLASS ) << "A^T*A=" << m_ata.rows() << "x" << m_ata.cols();

    m_isPrepared = true;
    return m_isPrepared;
}

bool WHPISignalExtraction::reconstructAmplitudes( WLEMDHPI::SPtr& hpiOut, WLEMDMEG::ConstSPtr megIn )
{
    wlog::debug( CLASS ) << "reconstructAmplitudes() called!";

    WLTimeProfiler prof( CLASS, __func__ );
    // Some error handling
    // -------------------
    if( !m_isPrepared )
    {
        wlog::warn( CLASS ) << "Algorithm is not prepared! Preparing it now.";
        if( !prepare() )
        {
            wlog::error( CLASS ) << "Could not prepare algorithm!";
            return false;
        }
    }
    if( megIn->getSampFreq() != m_sampFreq )
    {
        wlog::warn( CLASS ) << "Sampling frequencies are not equals! Preparing data with new sampling frequency.";
        setSamplingFrequency( m_sampFreq );
        if( !prepare() )
        {
            wlog::error( CLASS ) << "Could not re-prepare algorithm!";
            return false;
        }
    }
    if( m_at.rows() != static_cast< MatrixT::Index >( 2 * m_angFrequencies.size() ) )
    {
        wlog::error( CLASS ) << "Matrix A seems not to be prepared!";
        return false;
    }

    // Preparation
    // ----------
    const MatrixT::Index J = m_angFrequencies.size();
    const MatrixT::Index N = m_windowsSize * m_sampFreq; // windows size in samples
    const MatrixT::Index S = m_stepSize * m_sampFreq; // step size in samples

    const WLEMData::DataT::Index channels = static_cast< WLEMData::DataT::Index >( megIn->getNrChans() );
    const WLEMData::DataT::Index hpiChannels = channels * J;
    const WLEMData::DataT::Index samples = static_cast< WLEMData::DataT::Index >( megIn->getSamplesPerChan() );

    WLEMDHPI::DataSPtr dataOut( new WLEMDHPI::DataT( hpiChannels, samples / S ) );
    const WLEMData::DataT& data = megIn->getData();

    // Processing: Move windows step by step
    // -------------------------------------
    WLEMData::DataT::Index hpiSmp = 0;
    WLEMData::DataT blockPreproc( channels, N );
    WLEMData::SampleT hpiSample( hpiChannels );
    for( MatrixT::Index start = 0; start + N <= samples; start += S )
    {
        preprocessBlock( &blockPreproc, data.block( 0, start, channels, N ) );
        reconstructWindows( &hpiSample, blockPreproc );
        dataOut->block( 0, hpiSmp++, hpiChannels, 1 ) = hpiSample;
    }

    // Finalization
    // ------------
    if( !hpiOut )
    {
        hpiOut.reset( new WLEMDHPI );
    }
    hpiOut->setData( dataOut );
    hpiOut->setNrHpiCoils( J );
    hpiOut->setSampFreq( 1.0 / m_stepSize );

    return true;
}

void WHPISignalExtraction::preprocessBlock( WLEMData::DataT* const megOut, const WLEMData::DataT& megIn )
{
    const WLSampleNrT samples = megIn.cols();
    VectorT win = WLWindowFunction::hamming( samples );
    VectorT det( samples );
    for( WLEMDMEG::DataT::Index c = 0; c < megIn.rows(); ++c )
    {
        WLPreprocessing::detrend( &det, megIn.row( c ).transpose() );
        megOut->row( c ) = win.cwiseProduct( det );
    }
}

void WHPISignalExtraction::reconstructWindows( WLEMData::SampleT* const hpiOut, const WLEMData::DataT& megIn )
{
    const MatrixT::Index J = m_angFrequencies.size();
    MatrixT::Index hpiOffset = 0;

    for( WLEMDMEG::DataT::Index c = 0; c < megIn.rows(); ++c )
    {
        const VectorT& b = megIn.row( c );
        const VectorT atb = m_at * b;

        // Solve: x = (A^T*A)^-1 * A^T*b
        VectorT x = m_ata.colPivHouseholderQr().solve( atb );

        // x = x^2
        x = x.cwiseProduct( x );
        // Reduction: tmp = x'^2 + x''^2
        for( MatrixT::Index j = 0; j < J; ++j )
        {
            x( j ) += x( j + J );
        }
        // a = sqrt(x'^2 + x''^2)
        const VectorT a = x.segment( 0, J ).cwiseSqrt();
        hpiOut->block( hpiOffset, 0, J, 1 ) = a;

        hpiOffset += J;
    }
}

void WHPISignalExtraction::reset()
{
    wlog::debug( CLASS ) << "reset() called!";

    m_isPrepared = false;
    m_windowsSize = WINDOWS_SIZE;
    m_stepSize = STEP_SIZE;
    m_sampFreq = SAMPLING_FREQ;
    m_angFrequencies.clear();

    wlog::debug( CLASS ) << "Algorithm reset.";
}
