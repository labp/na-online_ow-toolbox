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

#include <cmath>

#include <core/common/WLogger.h>
#include "core/util/profiler/WLTimeProfiler.h"

#include "WHPISignalExtraction.h"

const std::string WHPISignalExtraction::CLASS = "WHPISignalExtraction";

static const WLTimeT WINDOWS_SIZE = 0.2;
static const WLTimeT STEP_SIZE = 0.01;
static const WLFreqT SAMPLING_FREQ = 1000.0;
static const float SEC_TO_MS = 1000.0;

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
    return m_windowsSize * SEC_TO_MS;
}

void WHPISignalExtraction::setWindowsSize( WLTimeT winSize )
{
    m_windowsSize = winSize / SEC_TO_MS;
    m_isPrepared = false;
}

WLTimeT WHPISignalExtraction::getStepSize() const
{
    return m_stepSize * SEC_TO_MS;
}

void WHPISignalExtraction::setStepSize( WLTimeT stepSize )
{
    m_stepSize = stepSize / SEC_TO_MS;
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
    WLTimeProfiler prof( CLASS, "pepare" );

    if( m_isPrepared )
    {
        return m_isPrepared;
    }
    if( m_sampFreq < 0.0 )
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

    const WLTimeT T = 1 / m_sampFreq;
    wlog::debug( CLASS ) << "T=" << T;

    for( MatrixT::Index i = 0; i < N; ++i )
    {
        for( MatrixT::Index j = 0; j < J; ++j )
        {
            m_a( i, j ) = sin( m_angFrequencies[j] * i * T );
            m_a( i, j + J ) = cos( m_angFrequencies[j] * i * T );
        }
    }
    wlog::debug( CLASS ) << "A=" << m_a.rows() << "x" << m_a.cols();

    m_at = m_a.transpose();
    m_ata = m_at * m_a;
    wlog::debug( CLASS ) << "A^T*A=" << m_ata.rows() << "x" << m_ata.cols();

    m_isPrepared = true;
    return m_isPrepared;
}

bool WHPISignalExtraction::reconstructAmplitudes( WLEMDHPI::SPtr hpiOut, WLEMDMEG::ConstSPtr megIn )
{
    wlog::debug( CLASS ) << "reconstructAmplitudes() called!";

    WLTimeProfiler prof( CLASS, "reconstructAmplitudes" );
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
    if( !m_lastMeg )
    {
        m_lastMeg = megIn;
        wlog::debug( CLASS ) << "No previous data. Reconstruction starts with next block.";
        return false;
    }
    if( m_at.rows() != static_cast< MatrixT::Index >( 2 * m_angFrequencies.size() ) )
    {
        wlog::error( CLASS ) << "Matrix A seems not to be prepared!";
        return false;
    }

    // Preparation: constants, variables, data
    // ---------------------------------------
    const MatrixT::Index J = m_angFrequencies.size();
    const MatrixT::Index N = m_windowsSize * m_sampFreq; // windows size in samples
    const MatrixT::Index S = m_stepSize * m_sampFreq; // step size in samples

    const WLEMData::DataT::Index channels = static_cast< WLEMData::DataT::Index >( megIn->getNrChans() );
    const WLEMData::DataT::Index hpiChannels = channels * J;
    const WLEMData::DataT::Index samples = static_cast< WLEMData::DataT::Index >( megIn->getSamplesPerChan() );

    // Prepare output data
    WLEMDHPI::DataSPtr dataOut( new WLEMDHPI::DataT( hpiChannels, samples / S ) );

    // Combine last block with current block
    WLEMData::DataT data( channels, 2 * samples );
    data.block( 0, 0, channels, samples ) = m_lastMeg->getData();
    data.block( 0, samples, channels, samples ) = megIn->getData();

    // Processing: Move windows step by step
    // -------------------------------------
    // TODO (pieloth): min(m_angFrequencies)/2 highpass filter is needed!

    WLEMData::DataT::Index hpiSmp = 0;
    for( MatrixT::Index start = samples - N; start + N < 2 * samples; start += S )
    {
        WLEMData::SampleT hpiSampel( hpiChannels );
        reconstructWindows( &hpiSampel, data, start, N );
        dataOut->block( 0, hpiSmp++, hpiChannels, 1 ) = hpiSampel;
    }

    // Finalization
    // ------------
    m_lastMeg = megIn;
    hpiOut->setData( dataOut );
    hpiOut->setNrHpiCoils( J );
    hpiOut->setSampFreq( 1.0 / m_stepSize );

    return true;
}

void WHPISignalExtraction::reconstructWindows( WLEMData::SampleT* const hpiOut, const WLEMData::DataT& megIn,
                MatrixT::Index start, MatrixT::Index samples )
{
    const MatrixT::Index J = m_angFrequencies.size();
    MatrixT::Index hpiOffset = 0;

    for( WLEMDMEG::DataT::Index c = 0; c < megIn.rows(); ++c )
    {
        const VectorT b = megIn.row( c ).segment( start, samples ).transpose();
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
    m_lastMeg.reset();
    m_angFrequencies.clear();

    wlog::debug( CLASS ) << "Algorithm reset.";
}
