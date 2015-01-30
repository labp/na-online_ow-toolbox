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
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include "core/util/profiler/WLTimeProfiler.h"

#include "core/data/WLMegCoilInfo.h"
#include "WMegForwardSphere.h"

typedef Eigen::Vector3d PositionT;
typedef Eigen::Vector3d OrientationT;
typedef Eigen::VectorXd VectorT;
typedef Eigen::Vector3d Vector3T;

static const double MY0 = 4 * M_PI * 1.E-7; //!< absolute permeability
static const double EPS = 0.0001;

const std::string WMegForwardSphere::CLASS = "WMegForward";

WMegForwardSphere::WMegForwardSphere()
{
}

WMegForwardSphere::~WMegForwardSphere()
{
}

void WMegForwardSphere::setMegCoilInfos( WLArrayList< WLMegCoilInfo::SPtr >::SPtr coilInfos )
{
    m_coilInfos = coilInfos;
    const WLArrayList< WLMegCoilInfo::SPtr >::size_type n_coils = m_coilInfos->size();
    m_intPntDev.clear();
    m_intPntDev.reserve( n_coils );

    wlog::debug( CLASS ) << "Transform integration points to device coords.";
    for( WLArrayList< WLMegCoilInfo::SPtr >::size_type i = 0; i < n_coils; ++i )
    {
        WLMegCoilInfo::SPtr coilInfo = ( *m_coilInfos )[i];
        PositionsT intPnts( 3, coilInfo->integrationPoints.cols() );
        if( !transformIntPntLocal2Dev( &intPnts, *coilInfo ) )
        {
            m_intPntDev.clear();
            return;
        }
        m_intPntDev.push_back( intPnts );
    }

    wlog::debug( CLASS ) << "Calculate weber to tesla.";
}

bool WMegForwardSphere::transformIntPntLocal2Dev( PositionsT* ipOut, const WLMegCoilInfo& megCoilInfo )
{
    const PositionsT::Index n_intpnt = megCoilInfo.integrationPoints.cols();
    if( n_intpnt < 1 )
    {
        wlog::error( CLASS ) << __func__ << ": No integration points available!";
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

bool WMegForwardSphere::computeForward( MatrixT* const pLfOut, const PositionsT& dipPos, const OrientationsT& dipOri )
{
    WAssert( dipPos.cols() == dipOri.cols(), "#dipPos != #dipOri" );
    WLTimeProfiler profiler( CLASS, __func__ );

    // Check some pre-conditions & prepare
    // -----------------------------------
    if( dipPos.cols() < 1 || m_intPntDev.empty() )
    {
        wlog::error( CLASS ) << "No dipoles or integration points!";
        return false;
    }
    if( !m_coilInfos || m_coilInfos->empty() )
    {
        wlog::error( CLASS ) << "No coil information!";
        return false;
    }

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

    const int n_sensors = m_coilInfos->size();
    const int n_dips = dipPos.cols();

    // Allocate Result Matrix
    lfOut.resize( n_sensors, n_dips );
    lfOut.setZero();

    // Compute forward model
    // ---------------------
    // Compute for each Dipole ...
    for( int iDip = 0; iDip < n_dips; ++iDip )
    {
        R0 = dipPos.col( iDip ) - cPos;
        Q = dipOri.col( iDip );

        // Compute for each Sensor
        for( int iSens = 0; iSens < n_sensors; ++iSens )
        {
            const WLMegCoilInfo& megCoilInfo = *m_coilInfos->at( iSens );
            const PositionsT& ip = m_intPntDev[iSens];
            const PositionsT::Index n_intpnt = ip.cols();
            WAssert( n_intpnt == megCoilInfo.integrationWeights.size(), "#ip != #ipWeights" );
            for( int iIntPnt = 0; iIntPnt < n_intpnt; ++iIntPnt )
            {
                if( fabs( megCoilInfo.integrationWeights( iIntPnt ) ) < EPS )
                {
                    wlog::debug( CLASS ) << __func__ << ": skip integrationWeight";
                    continue;
                }

                R = ip.col( iIntPnt ) - cPos;
                A = R - R0;

                r = R.norm();
                if( r < 1e-35 )
                {
                    wlog::error( CLASS ) << __func__ << ": r < 1e-35";
                    return false;
                } // gradiometer in origin

                a = A.norm();
                if( a < 1e-35 )
                {
                    wlog::error( CLASS ) << __func__ << ": a < 1e-35";
                    return false;
                } // dipole in gradiometer

                F = a * ( r * a + r * r - R0.dot( R ) );
                if( F < 1e-35 )
                {
                    wlog::error( CLASS ) << __func__ << ": F < 1e-35";
                    return false;
                }

                NablaF = ( a * a / r + A.dot( R ) / a + 2 * a + 2 * r ) * R;
                NablaF -= ( a + 2 * r + A.dot( R ) / a ) * R0;

                QxR0 = Q.cross( R0 );

                Aux1 = F * QxR0;
                Aux2 = QxR0.dot( R ) * NablaF;
                B = ( MY0 / ( 4 * M_PI * F * F ) ) * ( Aux1 - Aux2 );

                w = megCoilInfo.windings * megCoilInfo.integrationWeights( iIntPnt ) * megCoilInfo.area;
                const Vector3T mag_ori = megCoilInfo.orientation;
                lfOut( iSens, iDip ) += w * mag_ori.dot( B );
            } // for each integration point, TODO(pieloth) gradiometer???
            lfOut( iSens, iDip ) /= megCoilInfo.area; // Wb = Tm^2 --> Wb/m^2 = T
        }   // for each sensor
    }   // for each dipole

    return true;
}
