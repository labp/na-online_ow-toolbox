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

#include <cmath>  // sqrt, ceil
#include <limits> // max_double
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SparseLU>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/daqSystem/WLDaqNeuromag.h"
#include "core/util/WLGeometry.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WHeadPositionCorrection.h"
#include "WMegForwardSphere.h"

using Eigen::SparseLU;

typedef Eigen::SparseMatrix< ScalarT > SpMatrixT;

const std::string WHeadPositionCorrection::CLASS = "WHeadPositionCorrection";

WHeadPositionCorrection::WHeadPositionCorrection() :
                m_isInitialized( false ), m_movementThreshold( 0.001 ), m_radius( 0.07 )
{
    reset();
}

WHeadPositionCorrection::~WHeadPositionCorrection()
{
}

bool WHeadPositionCorrection::init()
{
    WLTimeProfiler profiler( CLASS, __func__, true );
    wlog::debug( CLASS ) << __func__ << "() called!";

    // TODO(pieloth): Check unit/exponent/from/to megPos, dipPos, refTrans

    // Check pre-conditions
    // --------------------
    if( m_isInitialized )
    {
        return true;
    }
    m_isInitialized = false;
    if( m_megPos.empty() || m_megOri.cols() == 0 )
    {
        wlog::error( CLASS ) << "MEG positions and orientations are not set!";
        return false;
    }
    if( !m_coilInfos )
    {
        wlog::error( CLASS ) << "MEG coil infos are not set!";
        return false;
    }
    if( m_megPos.coordSystem() != WLECoordSystem::UNKNOWN && m_megPos.coordSystem() != WLECoordSystem::DEVICE )
    {
        wlog::error( CLASS ) << "MEG coordinate system is wrong: " << m_megPos.coordSystem();
        return false;
    }
    if( m_megPos.unit() != WLEUnit::UNKNOWN && m_megPos.unit() != WLEUnit::METER )
    {
        wlog::error( CLASS ) << "Unit is not meter!";
        return false;

    }
    if( m_megPos.exponent() != WLEExponent::UNKNOWN && m_megPos.exponent() != WLEExponent::BASE )
    {
        wlog::error( CLASS ) << "Exponent is not meter!";
        return false;
    }
    if( m_transRef.from() != WLECoordSystem::HEAD || m_transRef.to() != WLECoordSystem::DEVICE )
    {
        wlog::error( CLASS ) << "Reference transformation has wrong from/to coordinate system!";
        return false;
    }

    // Initialize
    // ----------
    // 1. generate simples dipole sphere: positions, orientations
    const PositionsT::IndexT nDip = m_megPos.size() * 2;
    if( !generateDipoleSphere( &m_dipPos, &m_dipOri, nDip, m_radius ) )
    {
        wlog::error( CLASS ) << "Error on creating sphere.";
        return false;
    }
    m_dipPos.unit( m_megPos.unit() );
    m_dipPos.exponent( m_megPos.exponent() );
    m_dipPos.coordSystem( WLECoordSystem::HEAD );
    WAssertDebug( !m_dipPos.empty() && m_dipOri.cols() > 0, "Dipoles are emtpy." );

    // 2. align sphere to head model/coordinates
    // necessary?

    // 3. compute forward model for sphere at reference position
    PositionsT::SPtr dipPos;
    OrientationsT dipOriRef;
    // Move dipoles to reference position
    dipPos = m_transRef * m_dipPos;
    dipOriRef = m_transRef.rotation() * m_dipOri;
    if( !computeForward( &m_lfRef, m_megPos, m_megOri, *dipPos, dipOriRef ) )
    {
        wlog::error( CLASS ) << "Error on computing leadfield for reference position.";
        return false;
    }

    m_isInitialized = true;
    return true;
}

bool WHeadPositionCorrection::isInitialzied() const
{
    return m_isInitialized;
}

void WHeadPositionCorrection::reset()
{
    m_isInitialized = false;
    m_transRef.setIdentity();
}

bool WHeadPositionCorrection::process( WLEMDMEG* const megOut, const WLEMDMEG& megIn, const WLEMDHPI& hpi )
{
    WLTimeProfiler profiler( CLASS, __func__, true );
    if( !m_isInitialized && !init() )
    {
        wlog::error( CLASS ) << "Error on initializing.";
        return false;
    }

    const WLFreqT sfreq_hpi = hpi.getSampFreq();
    const WLFreqT sfreq_meg = megIn.getSampFreq();
    if( sfreq_hpi == WLEMData::UNDEFINED_FREQ || sfreq_hpi == WLEMData::UNDEFINED_FREQ )
    {
        wlog::error( CLASS ) << "Sampling frequency from HPI or MEG not set!";
        return false;
    }
    if( sfreq_hpi > sfreq_meg || static_cast< size_t >( sfreq_meg.value() ) % static_cast< size_t >( sfreq_hpi.value() ) != 0 )
    {
        wlog::error( CLASS ) << "Pre-conditions not hold: sfreq_hpi <= sfreq_meg AND sfreq_meg%sfreq_hpi != 0";
        return false;
    }

    WLEMData::DataT::Index smpOffset = sfreq_meg / sfreq_hpi;
    WAssert( smpOffset > 0, "Offset is less or equals 0!" );

    WLArrayList< WLEMDHPI::TransformationT >::ConstSPtr trans = hpi.getTransformations();
    WLArrayList< WLEMDHPI::TransformationT >::const_iterator itTrans;
    PositionsT::SPtr dipPos;
    OrientationsT dipOri;
    WLEMData::DataT::Index smpStart = 0;

    const WLChanNrT chans = megIn.getNrChans();
    const WLSampleNrT smpls = megIn.getSamplesPerChan();
    WLEMData::DataSPtr dataOut( new WLEMData::DataT( chans, smpls ) );
    megOut->setData( dataOut );
    for( itTrans = trans->begin(); itTrans != trans->end(); ++itTrans )
    {
        if( checkMovementThreshold( *itTrans ) )
        {
            WLTimeProfiler profilerTras( CLASS, "process_trans", true );
            // 4. transform dipole sphere to head position, MEG coord, in case head position has changed
            dipPos = *itTrans * m_dipPos;
            dipOri = itTrans->rotation() * m_dipOri;
            // 5. compute forward model+inverse operator, in case head position has changed
            if( !computeForward( &m_lfNow, m_megPos, m_megOri, *dipPos, dipOri ) )
            {
                wlog::error( CLASS ) << "Error on computing forward solution for current head position.";
                return false;
            }
            if( !computeInverseOperation( &m_gNow, m_lfNow ) )
            {
                wlog::error( CLASS ) << "Error on computing inverse operator for current head position.";
                return false;
            }
        }

        // 6. compute inverse solution
        // attention: apply/interpolate different sFreq in HPI and MEG data!
        WLEMData::DataT dipData = m_gNow * megIn.getData().block( 0, smpStart, chans, smpOffset );
        // 7. compute forward solution at reference position
        dataOut->block( 0, smpStart, chans, smpOffset ) = m_lfRef * dipData;

        smpStart += smpOffset;
    }

    WAssertDebug( smpStart >= smpStart, "smpStart >= smpStart" );
    WAssertDebug( megOut->getSamplesPerChan() == megIn.getSamplesPerChan(), "Sample size does not match." );
    WAssertDebug( megOut->getNrChans() == megIn.getNrChans(), "Channels size does not match." );
    return true;
}

void WHeadPositionCorrection::setRefTransformation( const WLTransformation& trans )
{
    if( m_transRef.data() != trans.data() )
    {
        m_transRef = trans;
        m_isInitialized = false;
    }
}

void WHeadPositionCorrection::setMegCoilInfos( WLArrayList< WLMegCoilInfo::SPtr >::SPtr coilInfo )
{
    WLTimeProfiler profiler( CLASS, __func__, true );
    m_isInitialized = false;
    const WLArrayList< WLMegCoilInfo::SPtr >::size_type n_coils = coilInfo->size();
    if( n_coils == 0 )
    {
        m_coilInfos.reset();
        return;
    }

    m_coilInfos = coilInfo;
    m_megPos.resize( n_coils );
    m_megPos.unit( WLEUnit::METER ); // Position from MEG coil infos are in meter, see API
    m_megPos.exponent( WLEExponent::BASE ); // Position from MEG coil infos are in meter, see API
    m_megPos.coordSystem( WLECoordSystem::DEVICE );
    m_megOri.resize( 3, n_coils );

    for( WLArrayList< WLMegCoilInfo::SPtr >::size_type i = 0; i < n_coils; ++i )
    {
        m_megPos.data().col( i ) = ( *m_coilInfos )[i]->position;
        m_megOri.col( i ) = ( *m_coilInfos )[i]->orientation;
    }
    applyCoilIntegrationPoints( m_coilInfos.get() );
}

void WHeadPositionCorrection::setSphereRadius( float r )
{
    WAssert( r < 0.3, "Sphere radius is to large." );
    m_radius = r;
}

void WHeadPositionCorrection::setMovementThreshold( float t )
{
    m_movementThreshold = t;
}

bool WHeadPositionCorrection::generateDipoleSphere( PositionsT* const pos, OrientationsT* const ori, size_t nDipoles,
                float r ) const
{
    WLTimeProfiler profiler( CLASS, __func__, true );

    // generate dipolse
    PositionsT pTmp;
    const size_t points = WLGeometry::createUpperHalfSphere( &pTmp.data(), std::ceil( nDipoles / 2.0 ), r );
    if( points == 0 )
    {
        wlog::error( CLASS ) << "Error on creating sphere!";
        return false;
    }
    wlog::debug( CLASS ) << "Number of dipoles: " << 2 * points;

    pos->data( pTmp.data() );
    *pos += pTmp;

    // generate orientations
    ori->resize( Eigen::NoChange, 2 * points );
    OrientationT o1, o2;
    for( OrientationsT::Index i = 0; i < points; ++i )
    {
        if( !WLGeometry::findTagentPlane( &o1, &o2, pos->at( i ) ) )
        {
            wlog::error( CLASS ) << "Error on creating orientations!";
            return false;
        }
        ori->col( i ) = o1 / o1.norm();
        ori->col( i + points ) = o2 / o2.norm();
    }

    WAssertDebug( pos->size() >= nDipoles, "Dipole count does not match." );
    WAssertDebug( pos->size() == ori->cols(), "Positions and orientation count does not match." );
    return true;
}

bool WHeadPositionCorrection::computeForward( MatrixT* const lf, const PositionsT& mPos, const OrientationsT& mOri,
                const PositionsT& dPos, const OrientationsT& dOri ) const
{
    WAssertDebug( mPos.size() == mOri.cols(), "Positions and orientation count does not match." );
    WAssertDebug( dPos.size() == dOri.cols(), "Dipole and orientation count does not match." );

    WLTimeProfiler profiler( CLASS, __func__, true );

    if( mPos.coordSystem() != dPos.coordSystem() )
    {
        wlog::error( CLASS ) << "MEG and dipole coordinate systems are not equals!";
        return false;
    }

    WMegForwardSphere megForward;
    megForward.setMegCoilInfos( m_coilInfos );
    if( !megForward.computeForward( lf, dPos.data(), dOri ) )
    {
        return false;
    }

    WAssertDebug( lf->rows() == mPos.size() && lf->cols() == dPos.size(),
                    "Dimension of L, MEG channels and dipoles does not match." );
    return true;
}

bool WHeadPositionCorrection::computeInverseOperation( MatrixT* const g, const MatrixT& lf ) const
{
    WLTimeProfiler profiler( CLASS, __func__, true );

    const float snr = 25;
    const MatrixT noiseCov = MatrixT::Identity( lf.rows(), lf.rows() );

    // Leafield transpose matrix
    const MatrixT LT = lf.transpose();

    // WinvLT = W^-1 * LT
    SpMatrixT w = SpMatrixT( lf.cols(), lf.cols() );
    w.setIdentity();
    w.makeCompressed();

    SparseLU< SpMatrixT > spSolver;
    spSolver.compute( w );
    if( spSolver.info() != Eigen::Success )
    {
        wlog::error( CLASS ) << "spSolver.compute( weighting ) not succeeded: " << spSolver.info();
        return false;
    }
    const MatrixT WinvLT = spSolver.solve( LT ); // needs dense matrix, returns dense matrix
    if( spSolver.info() != Eigen::Success )
    {
        wlog::error( CLASS ) << "spSolver.solve( LT ) not succeeded: " << spSolver.info();
        return false;
    }
    wlog::debug( CLASS ) << "WinvLT " << WinvLT.rows() << " x " << WinvLT.cols();

    // LWL = L * W^-1 * LT
    const MatrixT LWL = lf * WinvLT;
    wlog::debug( CLASS ) << "LWL " << LWL.rows() << " x " << LWL.cols();

    // alpha = sqrt(trace(LWL)/(snr * num_sensors));
    double alpha = sqrt( LWL.trace() / ( snr * lf.rows() ) );

    // G = W^-1 * LT * inv( (L W^-1 * LT) + alpha^2 * Cn )
    const MatrixT toInv = LWL + pow( alpha, 2 ) * noiseCov;
    const MatrixT inv = toInv.inverse();
    *g = WinvLT * inv;

    WAssertDebug( g->rows() == lf.cols() && g->cols() == lf.rows(), "Dimension of G and L does not match." );
    return true;
}

bool WHeadPositionCorrection::checkMovementThreshold( const WLEMDHPI::TransformationT& trans )
{
    const Eigen::Vector4d diffTrans = m_transExc.data().col( 3 ) - trans.data().col( 3 );
    if( diffTrans.norm() > m_movementThreshold )
    {
        m_transExc = trans;
        return true;
    }
    // TODO(pieloth): check rotation, e.g. with 1 or more key points
    // initial: pick 6 key points from dipoles min/max (x,0,0); (0,y,0); (0,0,z)
    // compare max distance
    // keyPoint 3x6
    // (trans * keyPoints).colwise().norm() - (m_transExc * keyPoints).colwise().norm()).maxCoeff(), mind homog. coord.
    return false;
}

void WHeadPositionCorrection::applyCoilIntegrationPoints( std::vector< WLMegCoilInfo::SPtr >* const coilInfos )
{
    const std::vector< WLMegCoilInfo::SPtr >::size_type n_coils = coilInfos->size();
    for( std::vector< WLMegCoilInfo::SPtr >::size_type i = 0; i < n_coils; ++i )
    {
        WLMegCoilInfo::SPtr coilInfo = ( *coilInfos )[i];
        // TODO(pieloth): Check correct coil type or read it in daq modules!
        if( i > 1 && ( i - 2 ) % 3 == 0 ) // magnetometer
        {
            WLDaqNeuromag::applyIntegrationPoints3022( coilInfo.get() );
        }
        else // gradiometer
        {
            WLDaqNeuromag::applyIntegrationPoints3012( coilInfo.get() );
        }
    }
}
