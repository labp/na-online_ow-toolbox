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

#include <algorithm> // transform
#include <cmath>
#include <functional> // minus
#include <set>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include <Eigen/SuperLUSupport>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "WSourceReconstruction.h"

using Eigen::SuperLU;
using Eigen::Triplet;
using std::minus;
using std::set;
using std::transform;

typedef Eigen::Triplet< double > TripletT;

const std::string WSourceReconstruction::CLASS = "WSourceReconstruction";

WSourceReconstruction::WSourceReconstruction()
{
}

WSourceReconstruction::~WSourceReconstruction()
{
}

void WSourceReconstruction::reset()
{
    m_leadfield.reset();
    m_weighting.reset();
    m_inverse.reset();
}

void WSourceReconstruction::setLeadfield( LaBP::MatrixSPtr matrix )
{
    m_leadfield = matrix;
    m_weighting.reset();
    m_inverse.reset();
}

const LaBP::MatrixT& WSourceReconstruction::getLeadfield() const
{
    return *m_leadfield;
}

bool WSourceReconstruction::hasLeadfield() const
{
    return m_leadfield.get() != NULL;
}

bool WSourceReconstruction::calculateWeightningMatrix( WSourceReconstruction::WEWeightingCalculation::Enum type )
{
    if( !m_leadfield )
    {
        wlog::error( CLASS ) << "No leadfield matrix available!";
        m_weighting.reset();
        return false;
    }

    switch( type )
    {
        case WEWeightingCalculation::MN:
        {
            const LaBP::SpMatrixT::Index wmRows = m_leadfield->cols();
            const LaBP::SpMatrixT::Index wmCols = m_leadfield->cols();
            m_weighting.reset( new LaBP::SpMatrixT( wmRows, wmCols ) );

            for( LaBP::SpMatrixT::Index i = 0; i < m_weighting->cols(); ++i )
            {
                m_weighting->insert( i, i ) = 1;
            }

            m_inverse.reset();
            return true;
        }
        case WEWeightingCalculation::WMN:
        {
            const LaBP::SpMatrixT::Index wmRows = m_leadfield->cols();
            const LaBP::SpMatrixT::Index wmCols = m_leadfield->cols();
            m_weighting.reset( new LaBP::SpMatrixT( wmRows, wmCols ) );

            const LaBP::MatrixT::Index lfRows = m_leadfield->rows();
            LaBP::MatrixElementT sum;
            for( LaBP::SpMatrixT::Index col = 0; col < wmRows; ++col )
            {
                sum = 0;
                for( LaBP::MatrixT::Index chan = 0; chan < lfRows; ++chan )
                {
                    sum += pow( ( *m_leadfield )( chan, col ), 2 );
                }
                m_weighting->insert( col, col ) = sqrt( sum );
            }
            // TODO(pieloth): Check weighted minimum norm.
            wlog::warn( CLASS ) << "WEIGHTED_MINIMUM_NORM still needs to be checked!";

            m_inverse.reset();
            return true;
        }
        default:
            wlog::error( CLASS ) << "Unknown weighting calculation!";
            m_weighting.reset();
            return false;
    }
}

const LaBP::SpMatrixT& WSourceReconstruction::getWeighting() const
{
    return *m_weighting;
}

bool WSourceReconstruction::hasWeighting() const
{
    return m_weighting.get() != NULL;
}

const LaBP::MatrixT& WSourceReconstruction::getInverse() const
{
    return *m_inverse;
}

bool WSourceReconstruction::hasInverse() const
{
    return m_inverse.get() != NULL;
}

bool WSourceReconstruction::calculateInverseSolution( const LaBP::MatrixT& noiseCov, const LaBP::MatrixT& dataCov, double snr )
{
    wlog::debug( CLASS ) << "calculateInverseSolution() called!";
    if( !m_leadfield )
    {
        wlog::error( CLASS ) << "No leadfield matrix set!";
        return false;
    }
    else
    {
        wlog::debug( CLASS ) << "m_leadfield " << m_leadfield->rows() << " x " << m_leadfield->cols();
    }

    if( !m_weighting )
    {
        wlog::error( CLASS ) << "No weighting matrix set!";
        return false;
    }
    else
    {
        wlog::debug( CLASS ) << "m_weighting " << m_weighting->rows() << " x " << m_weighting->cols();
    }

    wlog::debug( CLASS ) << "noiseCov " << noiseCov.rows() << " x " << noiseCov.cols();
    wlog::debug( CLASS ) << "dataCov " << dataCov.rows() << " x " << dataCov.cols();
    wlog::debug( CLASS ) << "snr " << snr;

    // Leafield transpose matrix
    LaBP::MatrixT LT = m_leadfield->transpose();
    wlog::debug( CLASS ) << "LT " << LT.rows() << " x " << LT.cols();

    // WinvLT = W^-1 * LT
    SuperLU< LaBP::SpMatrixT > spSolver;
    spSolver.compute( *m_weighting );
    if( spSolver.info() != Eigen::Success )
    {
        wlog::error( CLASS ) << "spSolver.compute( weighting ) not succeeded: " << spSolver.info();
        return false;
    }
    LaBP::MatrixT WinvLT = spSolver.solve( LT ); // needs dense matrix, returns dense matrix
    if( spSolver.info() != Eigen::Success )
    {
        wlog::error( CLASS ) << "spSolver.solve( LT ) not succeeded: " << spSolver.info();
        return false;
    }
    wlog::debug( CLASS ) << "WinvLT " << WinvLT.rows() << " x " << WinvLT.cols();

    // LWL = L * W^-1 * LT
    LaBP::MatrixT LWL = *m_leadfield * WinvLT;
    wlog::debug( CLASS ) << "LWL " << LWL.rows() << " x " << LWL.cols();

    // alpha = sqrt(trace(LWL)/(snr * num_sensors));
    double alpha = sqrt( LWL.trace() / ( snr * m_leadfield->rows() ) );
    wlog::debug( CLASS ) << "alpha " << alpha;

    // G = W^-1 * LT * inv( (L W^-1 * LT) + alpha^2 * Cn )
    LaBP::MatrixT toInv = LWL + pow( alpha, 2 ) * noiseCov;
    wlog::debug( CLASS ) << "toInv " << toInv.rows() << " x " << toInv.cols();

    LaBP::MatrixT inv = toInv.inverse();

    LaBP::MatrixT G = WinvLT * inv;
    m_inverse.reset( new LaBP::MatrixT( G ) );
    wlog::debug( CLASS ) << "G " << G.rows() << " x " << G.cols();

    return true;
}

WLEMDSource::SPtr WSourceReconstruction::createEMDSource( WLEMData::ConstSPtr emd,
                const LaBP::MatrixT matrix )
{
    boost::shared_ptr< WLEMData::DataT > data = WLEMDSource::convertMatrix( matrix );
    WLEMDSource::SPtr sourceEmd( new WLEMDSource( *emd ) );
    sourceEmd->setData( data );
    return sourceEmd;
}

bool WSourceReconstruction::averageReference( WLEMData::DataT& dataOut, const WLEMData::DataT& dataIn )
{
    wlog::debug( CLASS ) << "averageReference() called!";
    WLEMData::ChannelT dataSum( dataIn.front().size(), 0 );

    // calculate sum
    for( size_t chan = 0; chan < dataIn.size(); ++chan )
    {
        std::transform( dataIn[chan].begin(), dataIn[chan].end(), dataSum.begin(), dataSum.begin(),
                        std::plus< WLEMData::SampleT >() );
    }

    // calculate average
    const size_t count = dataIn.size();
    std::transform( dataSum.begin(), dataSum.end(), dataSum.begin(), std::bind2nd( std::divides< double >(), count ) );

    // calculate reference
    dataOut.resize( dataIn.size() );
    for( size_t chan = 0; chan < dataIn.size(); ++chan )
    {
        dataOut[chan].reserve( dataSum.size() );
        dataOut[chan].resize( dataSum.size() );

        std::transform( dataIn[chan].begin(), dataIn[chan].end(), dataSum.begin(), dataOut[chan].begin(),
                        std::minus< WLEMData::SampleT >() );
    }

    return true;
}

set< WSourceReconstruction::WEWeightingCalculation::Enum > WSourceReconstruction::WEWeightingCalculation::values()
{
    set< WSourceReconstruction::WEWeightingCalculation::Enum > values;
    values.insert( WEWeightingCalculation::MN );
    values.insert( WEWeightingCalculation::WMN );
    return values;
}

std::string WSourceReconstruction::WEWeightingCalculation::name( WSourceReconstruction::WEWeightingCalculation::Enum value )
{
    switch( value )
    {
        case WEWeightingCalculation::MN:
            return "MN";
        case WEWeightingCalculation::WMN:
            return "WMN";
        default:
            WAssert( false, "Unknown WEWeightingCalculation!" );
            return LaBP::UNDEFINED;
    }
}
