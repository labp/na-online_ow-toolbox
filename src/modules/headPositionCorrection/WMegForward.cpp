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
#include <limits>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMegCoilInformation.h"
#include "WMegForward.h"

static const double MY0 = 4 * M_PI * 1.E-7; //!< absolute permeability
static const double EPS = 0.0001;

#define ACCESS_3D( a, dy, dz, x, y ,z) (a[x * dy * dz + y * dz + z])

double WMegForward::weberToTesla( const WMegCoilInformation::WMegCoils& megSensor )
{
    double N = 0.0;
    const int nNumberOfCoils = megSensor.positions.cols();

    // a) find coil with lowest z amongst those with positive sense
    double MinZ = std::numeric_limits< double >::max();
    for( int i = 0; i < nNumberOfCoils; i++ )
    {
        if( megSensor.windings( i ) < 0 )
        {
            if( megSensor.positions( 2, i ) < MinZ )
            {
                MinZ = megSensor.positions( 2, i ); // TODO(pieloth) absolute value?
            }
        }
    }

    // b) sum up numbers of windings of all coils with negative sense and z smaller than a)
    for( int i = 0; i < nNumberOfCoils; i++ )
        if( megSensor.windings( i ) > 0 && megSensor.positions( 2, i ) < MinZ )
            N += megSensor.windings( i ) * megSensor.areas( i );

    // c) if N=0 negative senses are summed up
    if( !N )
        for( int i = 0; i < nNumberOfCoils; ++i )
            N += fabs( megSensor.windings( i ) * megSensor.areas( i ) );

    return N;
}

bool WMegForward::computeIntegrationPoints( PositionsT* ipOut, const WMegCoilInformation::WMegCoil& megCoilInfo )
{
    if( ipOut == NULL )
    {
        wlog::error( NSNAME ) << __func__ << ": ipOut is NULL!";
        return false;
    }

    PositionsT::Index n_intpnt = megCoilInfo.integrationPoints.cols();
    if( n_intpnt < 1 )
    {
        wlog::error( NSNAME ) << __func__ << ": No integration points available!";
        return false;
    }
    ipOut->resize( 3, megCoilInfo.integrationPoints.cols() );

    // T = [ex 0; ey 0; ez 0; p 1]'
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block( 0, 0, 3, 1 ) = megCoilInfo.ex;
    T.block( 0, 1, 3, 1 ) = megCoilInfo.ey;
    T.block( 0, 2, 3, 1 ) = megCoilInfo.ez;
    T.block( 0, 3, 3, 1 ) = megCoilInfo.position;

    // ip' = T * ip ... ip as homogeneous point
    ipOut->block( 0, 0, 3, n_intpnt ) = ( T * megCoilInfo.integrationPoints.colwise().homogeneous() ).block( 0, 0, 3, n_intpnt );

    return true;
}

bool WMegForward::computeForward( MatrixT* const pLfOut, const WMegCoilInformation::WMegCoils& megSensors,
                const PositionsT& dipPos, const OrientationsT& dipOri )
{
    WAssertDebug( megSensors.positions.cols() == megSensors.orientations.cols(), "#pos != #ori" );
    WAssertDebug( megSensors.positions.cols() == megSensors.windings.size(), "#pos != #windings" );
    WAssertDebug( megSensors.positions.cols() == megSensors.areas.size(), "#pos != #areas" );
    WAssertDebug( dipPos.cols() == dipOri.cols(), "#dipPos != #dipOri" );
    WLTimeProfiler profiler( NSNAME, __func__, true );

    const PositionT cPos = PositionT::Zero(); // TODO(pieloth): Use a passed in center point.
    MatrixT& lfOut = *pLfOut;

    Vector3T B;
    Vector3T R;                 //  Integration point related to center of sphere
    Vector3T R0;                //  Dipole position related to center of sphere
    Vector3T A;                 //  R-R0
    Vector3T Q;                 //  Dipole moment (direction)
    Vector3T QxR0;
    Vector3T NablaF;
    Vector3T Aux1;              //  auxiliary variable
    Vector3T Aux2;              //  auxiliary variable
    double F, a, r, w;

    const int n_sensors = megSensors.positions.cols();
    const int n_intpnt = megSensors.integrationPoints.cols();
    const int n_dips = dipPos.cols();

    // Allocate Result Matrix
    lfOut.resize( n_sensors, n_dips );
    lfOut.setZero();

    // Compute for each Dipole ...
    for( int iDip = 0; iDip < n_dips; ++iDip )
    {
        R0 = dipPos.col( iDip ) - cPos;
        Q = dipOri.col( iDip );

        // Compute for each Sensor
        for( int iSens = 0; iSens < n_sensors; ++iSens )
        {
            PositionsT ip(3, n_intpnt);
            // TODO
            WMegCoilInformation::WMegCoil megCoilInfo;
            megCoilInfo.ex.setRandom();
            megCoilInfo.ey.setRandom();
            megCoilInfo.ez.setRandom();
            megCoilInfo.integrationPoints = megSensors.integrationPoints;
            megCoilInfo.position = megSensors.positions.col(iSens);
            megCoilInfo.orientation = megSensors.orientations.col(iSens);
            if( !computeIntegrationPoints( &ip, megCoilInfo ) )
            {
                wlog::error( NSNAME ) << __func__ << ": !computeIntegrationPoints()";
                return false;
            }

            WAssert( n_intpnt == megSensors.integrationWeights.size(), "#ip != #ipWeights" );
            for( int iIntPnt = 0; iIntPnt < n_intpnt; ++iIntPnt )
            {
                if( fabs( megSensors.integrationWeights( iIntPnt ) ) < EPS )
                {
                    wlog::debug( NSNAME ) << __func__ << ": skip integrationWeight";
                    continue;
                }

                R = ip.col(iIntPnt) -cPos;
                A = R - R0;

                r = R.norm();
                if( r < 1e-35 )
                {
                    wlog::error( NSNAME ) << __func__ << ": r < 1e-35";
                    return false;
                } // gradiometer in origin

                a = A.norm();
                if( a < 1e-35 )
                {
                    wlog::error( NSNAME ) << __func__ << ": a < 1e-35";
                    return false;
                } // dipole in gradiometer

                F = a * ( r * a + r * r - R0.dot( R ) );
                if( F < 1e-35 )
                {
                    wlog::error( NSNAME ) << __func__ << ": F < 1e-35";
                    return false;
                }

                NablaF = ( a * a / r + A.dot( R ) / a + 2 * a + 2 * r ) * R;
                NablaF -= ( a + 2 * r + A.dot( R ) / a ) * R0;

                QxR0 = Q.cross( R0 );

                Aux1 = F * QxR0;
                Aux2 = QxR0.dot( R ) * NablaF;
                B = ( MY0 / ( 4 * M_PI * F * F ) ) * ( Aux1 - Aux2 );

                w = megSensors.windings( iSens ) * megSensors.integrationWeights( iIntPnt ) * megSensors.areas( iSens );
                const Vector3T mag_ori = megSensors.orientations.col( iSens );
                lfOut( iSens, iDip ) += w * mag_ori.dot( B );
            } // for each integration point, TODO(pieloth) gradiometer???
        }   // for each sensor
    }   // for each dipole
    lfOut /= weberToTesla( megSensors );
    return true;
}
