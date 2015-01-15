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

#include "WMegForward.h"
#include "WHeadPositionCorrection.h"

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

    // Check pre-conditions
    // --------------------
    if( m_isInitialized )
    {
        return true;
    }
    m_isInitialized = false;
    if( m_megPos.cols() == 0 || m_megOri.cols() == 0 )
    {
        wlog::error( CLASS ) << "MEG positions and orientations are not set!";
        return false;
    }

    // Initialize
    // ----------
    // 1. generate simples dipole sphere: positions, orientations
    const size_t nDip = m_megPos.cols() * 2;
    if( !generateDipoleSphere( &m_dipPos, &m_dipOri, nDip, m_radius ) )
    {
        wlog::error( CLASS ) << "Error on creating sphere.";
        return false;
    }
    WAssertDebug( m_dipPos.cols() > 0 && m_dipOri.cols() > 0, "Dipoles are emtpy." );

    // 2. align sphere to head model/coordinates
    // necessary?

    // 3. compute forward model for sphere at reference position
    PositionsT dipPosRef;
    OrientationsT dipOriRef;
    // Move dipoles to reference position
    dipPosRef = m_transRef * m_dipPos;
    dipOriRef = m_transRef.linear() * m_dipOri;
    if( !computeForward( &m_lfRef, m_megPos, m_megOri, dipPosRef, dipOriRef ) )
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
    if( sfreq_hpi > sfreq_meg || static_cast< size_t >( sfreq_meg ) % static_cast< size_t >( sfreq_hpi ) != 0 )
    {
        wlog::error( CLASS ) << "Pre-conditions not hold: sfreq_hpi <= sfreq_meg AND sfreq_meg%sfreq_hpi != 0";
        return false;
    }

    WLEMData::DataT::Index smpOffset = sfreq_meg / sfreq_hpi;
    WAssert( smpOffset > 0, "Offset is less or equals 0!" );

    WLArrayList< WLEMDHPI::TransformationT >::ConstSPtr trans = hpi.getTransformations();
    WLArrayList< WLEMDHPI::TransformationT >::const_iterator itTrans;
    PositionsT dipPos;
    OrientationsT dipOri;
    Eigen::Affine3d t;
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
            t = Eigen::Affine3d( *itTrans );
            dipPos = t * m_dipPos;
            dipOri = t.linear() * m_dipOri;  // attention: Eigen::Affine3d.rotation() returns a identity matrix!?
            // 5. compute forward model+inverse operator, in case head position has changed
            if( !computeForward( &m_lfNow, m_megPos, m_megOri, dipPos, dipOri ) )
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

void WHeadPositionCorrection::setRefTransformation( const WLMatrix4::Matrix4T& trans )
{
    if( m_transRef.matrix() != trans )
    {
        m_transRef.matrix() = trans;
        m_isInitialized = false;
    }
}

void WHeadPositionCorrection::setMegPosAndOri( const WLEMDMEG& meg )
{
    m_isInitialized = false;
    WLArrayList< WPosition >::ConstSPtr poss = meg.getChannelPositions3d();
    m_megPos.resize( 3, poss->size() );
    WLArrayList< WVector3f >::ConstSPtr oris = meg.getEz();
    m_megOri.resize( 3, oris->size() );

    WAssert( poss->size() == oris->size(), "Size of MEG positions and orientations does not match." );
    for( size_t i = 0; i < poss->size(); ++i )
    {
        m_megPos.col( i ).x() = poss->at( i ).x();
        m_megPos.col( i ).y() = poss->at( i ).y();
        m_megPos.col( i ).z() = poss->at( i ).z();

        m_megOri.col( i ).x() = oris->at( i ).x();
        m_megOri.col( i ).y() = oris->at( i ).y();
        m_megOri.col( i ).z() = oris->at( i ).z();
    }
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
    const size_t points = WLGeometry::createUpperHalfSphere( &pTmp, std::ceil( nDipoles / 2.0 ), r );
    if( points == 0 )
    {
        wlog::error( CLASS ) << "Error on creating sphere!";
        return false;
    }
    wlog::debug( CLASS ) << "Number of dipoles: " << 2 * points;

    pos->resize( Eigen::NoChange, 2 * points );
    pos->block( 0, 0, 3, points ) = pTmp;
    pos->block( 0, points, 3, points ) = pTmp;

    // generate orientations
    ori->resize( Eigen::NoChange, 2 * points );
    OrientationT o1, o2;
    for( OrientationsT::Index i = 0; i < points; ++i )
    {
        if( !WLGeometry::findTagentPlane( &o1, &o2, pos->col( i ) ) )
        {
            wlog::error( CLASS ) << "Error on creating orientations!";
            return false;
        }
        ori->col( i ) = o1 / o1.norm();
        ori->col( i + points ) = o2 / o2.norm();
    }

    WAssertDebug( pos->cols() >= nDipoles, "Dipole count does not match." );
    WAssertDebug( pos->rows() == ori->rows() && pos->cols() == ori->cols(), "Dimension of pos and ori does not match." );
    return true;
}

bool WHeadPositionCorrection::computeForward( MatrixT* const lf, const PositionsT& mPos, const OrientationsT& mOri,
                const PositionsT& dPos, const OrientationsT& dOri ) const
{
    WAssertDebug( mPos.rows() == mOri.rows() && mPos.cols() == mOri.cols(), "Dimension of MEG pos and ori does not match." );
    WAssertDebug( dPos.rows() == dOri.rows() && dPos.cols() == dOri.cols(), "Dimension of dipole pos and ori does not match." );

    WLTimeProfiler profiler( CLASS, __func__, true );

    // TODO(pieloth): Check correct coil type and differentiate between mag and grad. Split grad to 2 mags ...
    // TODO(pieloth): MEG/coil does not change, so do not generate it everytime.
    std::vector< WLMegCoilInfo::SPtr > coilInfos;
    createCoilInfos( &coilInfos, mPos, mOri );
    if( !WMegForward::computeForward( lf, coilInfos/*megSensors*/, dPos, dOri ) )
    {
        return false;
    }

    WAssertDebug( lf->rows() == mPos.cols() && lf->cols() == dPos.cols(),
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
    const Eigen::Vector4d diffTrans = m_transExc.col( 3 ) - trans.col( 3 );
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

void WHeadPositionCorrection::createCoilInfos( std::vector< WLMegCoilInfo::SPtr >* const coilInfos, const PositionsT& mPos,
                const OrientationsT& mOri )
{
    WLTimeProfiler profiler( CLASS, __func__, true );
    WAssertDebug( mPos.rows() == mOri.rows() && mPos.cols() == mOri.cols(), "Dimension of MEG pos and ori does not match." );

    coilInfos->reserve( mPos.cols() );
    for( PositionsT::Index i = 0; i < mPos.cols(); ++i )
    {
        WLMegCoilInfo::SPtr coilInfo( new WLMegCoilInfo() );
        coilInfo->position = mPos.col( i );
        coilInfo->orientation = mOri.col( i );
        // TODO(pieloth): ex, ey, ez
        coilInfo->ex.setRandom();
        coilInfo->ey.setRandom();
        coilInfo->ez.setRandom();
        if( i > 1 && ( i - 2 ) % 3 == 0 ) // magnetometer
        {
            WLDaqNeuromag::applyIntegrationPoints3022( coilInfo.get() );
        }
        else // gradiometer
        {
            WLDaqNeuromag::applyIntegrationPoints3012( coilInfo.get() );
        }
        coilInfos->push_back( coilInfo );
    }
}
