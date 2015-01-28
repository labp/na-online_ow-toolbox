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

#include <cmath> // PI, sin, cos

#include <Eigen/Geometry>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "WContinuousPositionEstimation.h"

const std::string WContinuousPositionEstimation::CLASS = "WContinuousPositionEstimation";

WContinuousPositionEstimation::WContinuousPositionEstimation( const WLPositions& hpiPos, const WLPositions& sensPos,
                const std::vector< WVector3f >& sensOri ) :
                DownhillSimplexMethod(), m_unit( hpiPos.unit() ), m_exponent( hpiPos.exponent() )
{
    WAssert( hpiPos.unit() == sensPos.unit(), "Units are not equals!" );
    WAssert( hpiPos.unit() == WLEUnit::UNKNOWN || hpiPos.unit() == WLEUnit::METER, "Unit is not meter!" );
    WAssert( hpiPos.exponent() == sensPos.exponent(), "Exponents are not equals!" );
    WAssert( hpiPos.exponent() == WLEExponent::UNKNOWN || hpiPos.exponent() == WLEExponent::BASE, "Is not base unit!" );
    WAssert( hpiPos.coordSystem() == WLECoordSystem::HEAD, "HPI positions are not in head coordinates!" );
    WAssert( sensPos.coordSystem() == WLECoordSystem::DEVICE, "Senor positions are not in device coordinates!" );

    if( hpiPos.unit() == WLEUnit::UNKNOWN || hpiPos.exponent() == WLEExponent::UNKNOWN )
    {
        wlog::warn( CLASS ) << "Unit or exponent is unknown, meter is used!";
    }

    // Transform HPI coil positions
    m_hpiPos = hpiPos.data().colwise().homogeneous();

    // Transform sensor positions
    m_sensPos.reserve( sensPos.size() );
    for( WLPositions::IndexT iPos = 0; iPos < sensPos.size(); ++iPos )
    {
        m_sensPos.push_back( PointT( sensPos.at( iPos ) ) );
    }

    // Transform sensor orientations
    m_sensOri.reserve( sensOri.size() );
    std::vector< WVector3f >::const_iterator itOri;
    for( itOri = sensOri.begin(); itOri != sensOri.end(); ++itOri )
    {
        m_sensOri.push_back( OrientationT( itOri->x(), itOri->y(), itOri->z() ) );
    }
    m_smpIdx = 0;
}

WContinuousPositionEstimation::~WContinuousPositionEstimation()
{
}

double WContinuousPositionEstimation::func( const ParamsT& x ) const
{
    const TransformationT trans = paramsToTrans( x );
    const HPointsT hpi = trans * m_hpiPos;

    const PointT hpi1( hpi.block( 0, 0, 3, 1 ) );
    const PointT hpi2( hpi.block( 0, 1, 3, 1 ) );
    const PointT hpi3( hpi.block( 0, 2, 3, 1 ) );
    const PointT hpi4( hpi.block( 0, 3, 3, 1 ) );
    const PointT hpi5( hpi.block( 0, 4, 3, 1 ) );

    MatrixT lf1 = computeLeadfield( hpi1, m_sensPos, m_sensOri );
    MatrixT lf2 = computeLeadfield( hpi2, m_sensPos, m_sensOri );
    MatrixT lf3 = computeLeadfield( hpi3, m_sensPos, m_sensOri );
    MatrixT lf4 = computeLeadfield( hpi4, m_sensPos, m_sensOri );
    MatrixT lf5 = computeLeadfield( hpi5, m_sensPos, m_sensOri );

    const MatrixT data1 = getSample( 0 );
    const MatrixT data2 = getSample( 1 );
    const MatrixT data3 = getSample( 2 );
    const MatrixT data4 = getSample( 3 );
    const MatrixT data5 = getSample( 4 );

    const MomentT mom1 = lf1.colPivHouseholderQr().solve( data1 );
    const MomentT mom2 = lf2.colPivHouseholderQr().solve( data2 );
    const MomentT mom3 = lf3.colPivHouseholderQr().solve( data3 );
    const MomentT mom4 = lf4.colPivHouseholderQr().solve( data4 );
    const MomentT mom5 = lf5.colPivHouseholderQr().solve( data5 );

    const MatrixT dif1 = data1 - ( lf1 * mom1 ).cwiseAbs();
    const MatrixT dif2 = data2 - ( lf2 * mom2 ).cwiseAbs();
    const MatrixT dif3 = data3 - ( lf3 * mom3 ).cwiseAbs();
    const MatrixT dif4 = data4 - ( lf4 * mom4 ).cwiseAbs();
    const MatrixT dif5 = data5 - ( lf5 * mom5 ).cwiseAbs();

    const double difNorm = dif1.squaredNorm() + dif2.squaredNorm() + dif3.squaredNorm() + dif4.squaredNorm() + dif5.squaredNorm();
    const double dataNorm = data1.squaredNorm() + data2.squaredNorm() + data3.squaredNorm() + data4.squaredNorm()
                    + data5.squaredNorm();

//    const double error = dif.squaredNorm() / data.squaredNorm();
    const double error = difNorm / dataNorm;
    return error;
}

void WContinuousPositionEstimation::setData( const MatrixT& data )
{
    m_smpIdx = 0;
    m_data = data;
}

void WContinuousPositionEstimation::nextSample()
{
    ++m_smpIdx;
    if( m_smpIdx >= m_data.cols() )
    {
        m_smpIdx = 0;
    }
}

WContinuousPositionEstimation::MatrixT WContinuousPositionEstimation::getSample( size_t coilIdx ) const
{
    MatrixT hpiData4Coil( m_data.rows() / 5, m_data.cols() );
    MatrixT::Index i = coilIdx;
    WContinuousPositionEstimation::MatrixT::Index j = 0;
    while( i < m_data.rows() )
    {
        hpiData4Coil.row( j ) = m_data.row( i );
        i += 5;
        ++j;
    }
    return hpiData4Coil.col( m_smpIdx );
}

WContinuousPositionEstimation::MatrixT WContinuousPositionEstimation::computeLeadfield( const PointT& dipPos,
                const std::vector< PointT >& sensPos, const std::vector< OrientationT >& sensOri )
{
    MatrixT lf( sensPos.size(), 3 );

    for( size_t i = 0; i < sensPos.size(); ++i )
    {
        lf.row( i ) = computeMagneticDipole( dipPos, sensPos[i], sensOri[i] );
    }

    return lf;
}

WContinuousPositionEstimation::Vector3T WContinuousPositionEstimation::computeMagneticDipole( const PointT& dipPos,
                const PointT& sensPos, const OrientationT& sensOri )
{
    // NOTE: Implementation from FieldTrip
    // Move coordinate system to the dipole source.
    const PointT r = sensPos - dipPos;

    const double r2 = pow( r.norm(), 2 );
    const double r5 = pow( r.norm(), 5 );

    const MomentT mx( 1, 0, 0 );
    const MomentT my( 0, 1, 0 );
    const MomentT mz( 0, 0, 1 );

    const PointT rx( r.x(), r.x(), r.x() );
    const PointT ry( r.y(), r.y(), r.y() );
    const PointT rz( r.z(), r.z(), r.z() );

    const Vector3T tx = 3.0 * rx.cwiseProduct( r ) - mx * r2;
    const Vector3T ty = 3.0 * ry.cwiseProduct( r ) - my * r2;
    const Vector3T tz = 3.0 * rz.cwiseProduct( r ) - mz * r2;

    Vector3T lf;
    lf.x() = tx.dot( sensOri );
    lf.y() = ty.dot( sensOri );
    lf.z() = tz.dot( sensOri );
    const double muo = 12.566370614e-7;
    lf = ( ( muo / ( 4 * M_PI ) ) * lf ) / r5;

    return lf;
}

WLPositions::SPtr WContinuousPositionEstimation::getResultPositions() const
{
    WLPositions::SPtr positions = WLPositions::instance();
    positions->unit( m_unit );
    positions->exponent( m_exponent );
    positions->coordSystem( WLECoordSystem::DEVICE );
    positions->resize( m_hpiPos.cols() );

    const HPointsT points = paramsToTrans( getResultParams() ) * m_hpiPos;
    for( HPointsT::Index i = 0; i < points.cols(); ++i )
    {
        positions->data().col( i ) = points.block( 0, i, 3, 1 );
    }

    return positions;
}

WLTransformation::SPtr WContinuousPositionEstimation::getResultTransformation() const
{
    WLTransformation::SPtr t = WLTransformation::instance();
    t->unit( m_unit );
    t->exponent( m_exponent );
    t->from( WLECoordSystem::HEAD );
    t->to( WLECoordSystem::DEVICE );
    t->data( paramsToTrans( getResultParams() ) );
    return t;
}

WContinuousPositionEstimation::TransformationT WContinuousPositionEstimation::paramsToTrans( const ParamsT& params ) const
{
    TransformationT trans = TransformationT::Identity();
    trans.block( 0, 3, 3, 1 ) = params.block( 3, 0, 3, 1 );

    // the individual angles (in radians)
    const double rx = params( 0 );
    const double ry = params( 1 );
    const double rz = params( 2 );

    // precompute the sin/cos values of the angles
    const double sX = sin( rx );
    const double cX = cos( rx );
    const double sY = sin( ry );
    const double cY = cos( ry );
    const double sZ = sin( rz );
    const double cZ = cos( rz );

    // Rotate around z, y and then x
    // rot = Rx * Ry * Rz
    RotationT rot = RotationT::Zero();
    // row 1
    rot( 0, 0 ) = cZ * cY;
    rot( 0, 1 ) = -sZ * cY;
    rot( 0, 2 ) = sY;
    // row 2
    rot( 1, 0 ) = cZ * sY * sX + sZ * cX;
    rot( 1, 1 ) = -sZ * sY * sX + cZ * cX;
    rot( 1, 2 ) = -cY * sX;

    // row 3
    rot( 2, 0 ) = -cZ * sY * cX + sZ * sX;
    rot( 2, 1 ) = sZ * sY * cX + cZ * sX;
    rot( 2, 2 ) = cY * cX;

    trans.block( 0, 0, 3, 3 ) = rot.block( 0, 0, 3, 3 );
    return trans;
}
