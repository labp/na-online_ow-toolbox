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

#include <core/common/WLogger.h>

#include "WContinuousPositionEstimation.h"

const std::string WContinuousPositionEstimation::CLASS = "WContinuousPositionEstimation";

WContinuousPositionEstimation::WContinuousPositionEstimation( const std::vector< PositionT >& sensPos,
                const std::vector< Vector3T >& sensOri ) :
                m_sensPos( sensPos ), m_sensOri( sensOri )
{
}

WContinuousPositionEstimation::~WContinuousPositionEstimation()
{
}

double WContinuousPositionEstimation::func( const PointT& x ) const
{
    const Vector3T hpi1( x.block( 0, 0, 1, 3 ).transpose() );
    const Vector3T hpi2( x.block( 0, 3, 1, 3 ).transpose() );
    const Vector3T hpi3( x.block( 0, 6, 1, 3 ).transpose() );
    const Vector3T hpi4( x.block( 0, 9, 1, 3 ).transpose() );
    const Vector3T hpi5( x.block( 0, 12, 1, 3 ).transpose() );

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

    const Vector3T mom1 = lf1.colPivHouseholderQr().solve( data1 );
    const Vector3T mom2 = lf2.colPivHouseholderQr().solve( data2 );
    const Vector3T mom3 = lf3.colPivHouseholderQr().solve( data3 );
    const Vector3T mom4 = lf4.colPivHouseholderQr().solve( data4 );
    const Vector3T mom5 = lf5.colPivHouseholderQr().solve( data5 );

    const MatrixT dif1 = data1 - lf1 * mom1;
    const MatrixT dif2 = data2 - lf2 * mom2;
    const MatrixT dif3 = data3 - lf3 * mom3;
    const MatrixT dif4 = data4 - lf4 * mom4;
    const MatrixT dif5 = data5 - lf5 * mom5;
//    const Vector3T mom = lfinv * data;
//    const MatrixT dif = data - lf * mom;
    const double difNorm = dif1.squaredNorm() + dif2.squaredNorm() + dif3.squaredNorm() + dif4.squaredNorm() + dif5.squaredNorm();
    const double dataNorm = data1.squaredNorm() + data2.squaredNorm() + data3.squaredNorm() + data4.squaredNorm()
                    + data5.squaredNorm();

//    const double error = dif.squaredNorm() / data.squaredNorm();
    const double error = difNorm / dataNorm;
    return error;
}

void WContinuousPositionEstimation::setData( const MatrixT& data )
{
    m_data = data;
}

WContinuousPositionEstimation::MatrixT WContinuousPositionEstimation::getSample( size_t coilIdx ) const
{
    const MatrixT::Index s = 50; // TODO increment and set external
//    return m_data.col( s );
    MatrixT hpiData4Coil( m_data.rows() / 5, m_data.cols() );
    MatrixT::Index i = coilIdx;
    WContinuousPositionEstimation::MatrixT::Index j = 0;
    while( i < m_data.rows() )
    {
        hpiData4Coil.row( j ) = m_data.row( i );
        i += 5;
        ++j;
    }
    return hpiData4Coil.col( s );
}

WContinuousPositionEstimation::MatrixT WContinuousPositionEstimation::computeLeadfield( const PositionT& dipPos,
                const std::vector< PositionT >& sensPos, const std::vector< Vector3T >& sensOri )
{
    MatrixT lf( sensPos.size(), 3 );

    for( size_t i = 0; i < sensPos.size(); ++i )
    {
        lf.row( i ) = computeMagneticDipole( dipPos, sensPos[i], sensOri[i] );
    }

    return lf;
}

WContinuousPositionEstimation::Vector3T WContinuousPositionEstimation::computeMagneticDipole( const PositionT& dipPos,
                const PositionT& sensPos, const Vector3T& sensOri )
{
    // NOTE: Implementation from FieldTrip
    // TODO(pieloth): Check math
    // Move coordinate system to the dipole source.
    const PositionT r = sensPos - dipPos;
    const double r2 = pow( r.norm(), 2 );
    const double r5 = pow( r.norm(), 5 );

    const Vector3T mx( 1, 0, 0 );
    const Vector3T my( 0, 1, 0 );
    const Vector3T mz( 0, 0, 1 );

    const PositionT rx( r.x(), r.x(), r.x() );
    const PositionT ry( r.y(), r.y(), r.y() );
    const PositionT rz( r.z(), r.z(), r.z() );

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
