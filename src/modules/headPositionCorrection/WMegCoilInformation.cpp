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

#include "WMegCoilInformation.h"

void WMegCoilInformation::neuromagCoil3012( WMegCoils* const megCoils )
{
    // TODO(pieloth): Check assumption that we map 1 gradiometer to 2 magnetometer.
    // map 1 gradiometer to "2 magnetometer" by just using the integration points and different weights (+/-).
    const PositionsT::Index n_intpnt = 16;

    // Integration points
    megCoils->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8;

    // Integration weights
    // +- 1 / ( 4*16.2mm )
    megCoils->integrationWeights.resize( n_intpnt );
    const double iw1 = 1.0 / ( 4 * 16.2 );
    const double iw2 = -1.0 * iw1;

    // magnetometer 1
    // --------------
    // +- 5.89, +- 6.71, 0.3mm
    ip1 << 5.89, 6.71, 0.3;
    ip2 << 5.89, -6.71, 0.3;
    ip3 << -5.89, 6.71, 0.3;
    ip4 << -5.89, -6.71, 0.3;

    // +- 10.8, +- 6.71, 0.3mm
    ip5 << 10.8, 6.71, 0.3;
    ip6 << 10.8, -6.71, 0.3;
    ip7 << -10.8, 6.71, 0.3;
    ip8 << -10.8, -6.71, 0.3;

    megCoils->integrationPoints.col( 0 ) = ip1;
    megCoils->integrationPoints.col( 1 ) = ip2;
    megCoils->integrationPoints.col( 2 ) = ip3;
    megCoils->integrationPoints.col( 3 ) = ip4;
    megCoils->integrationPoints.col( 4 ) = ip5;
    megCoils->integrationPoints.col( 5 ) = ip6;
    megCoils->integrationPoints.col( 6 ) = ip7;
    megCoils->integrationPoints.col( 7 ) = ip8;

    megCoils->integrationWeights( 0 ) = iw1;
    megCoils->integrationWeights( 1 ) = iw1;
    megCoils->integrationWeights( 2 ) = iw1;
    megCoils->integrationWeights( 3 ) = iw1;
    megCoils->integrationWeights( 4 ) = iw1;
    megCoils->integrationWeights( 5 ) = iw1;
    megCoils->integrationWeights( 6 ) = iw1;
    megCoils->integrationWeights( 7 ) = iw1;

    // magnetometer 2
    // --------------
    // +- 5.89, +- 6.71, 0.3mm
    ip1 << 5.89, 6.71, 0.3;
    ip2 << 5.89, -6.71, 0.3;
    ip3 << -5.89, 6.71, 0.3;
    ip4 << -5.89, -6.71, 0.3;

    // +- 10.8, +- 6.71, 0.3mm
    ip5 << 10.8, 6.71, 0.3;
    ip6 << 10.8, -6.71, 0.3;
    ip7 << -10.8, 6.71, 0.3;
    ip8 << -10.8, -6.71, 0.3;

    megCoils->integrationPoints.col( 8 ) = ip1;
    megCoils->integrationPoints.col( 9 ) = ip2;
    megCoils->integrationPoints.col( 10 ) = ip3;
    megCoils->integrationPoints.col( 11 ) = ip4;
    megCoils->integrationPoints.col( 12 ) = ip5;
    megCoils->integrationPoints.col( 13 ) = ip6;
    megCoils->integrationPoints.col( 14 ) = ip7;
    megCoils->integrationPoints.col( 15 ) = ip8;

    megCoils->integrationWeights( 8 ) = iw2;
    megCoils->integrationWeights( 9 ) = iw2;
    megCoils->integrationWeights( 10 ) = iw2;
    megCoils->integrationWeights( 11 ) = iw2;
    megCoils->integrationWeights( 12 ) = iw2;
    megCoils->integrationWeights( 13 ) = iw2;
    megCoils->integrationWeights( 14 ) = iw2;
    megCoils->integrationWeights( 15 ) = iw2;
}

void WMegCoilInformation::neuromagCoil3014( WMegCoils* const megCoils )
{
    // 3012 and 3014 has the same geometry
    neuromagCoil3012( megCoils );
}

void WMegCoilInformation::neuromagCoil3022( WMegCoils* const megCoils )
{
    const PositionsT::Index n_intpnt = 9;
    // Integration points
    PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9;
    ip1 << 0.0, 0.0, 0.3;

    // +- 9.99, +- 9.9, 0.3mm
    ip2 << 9.99, 9.99, 0.3;
    ip3 << 9.99, -9.99, 0.3;
    ip4 << -9.99, 9.99, 0.3;
    ip5 << -9.99, -9.99, 0.3;

    //  0.0, +- 9.99, 0.3mm
    ip6 << 0.0, 9.99, 0.3;
    ip7 << 0.0, -9.99, 0.3;

    // +- 9.99, 0.0, 0.3mm
    ip8 << 9.99, 0.0, 0.3;
    ip9 << -9.99, 0.0, 0.3;

    megCoils->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    megCoils->integrationPoints.col( 0 ) = ip1;
    megCoils->integrationPoints.col( 1 ) = ip2;
    megCoils->integrationPoints.col( 2 ) = ip3;
    megCoils->integrationPoints.col( 3 ) = ip4;
    megCoils->integrationPoints.col( 4 ) = ip5;
    megCoils->integrationPoints.col( 5 ) = ip6;
    megCoils->integrationPoints.col( 6 ) = ip7;
    megCoils->integrationPoints.col( 7 ) = ip8;
    megCoils->integrationPoints.col( 8 ) = ip9;

    // Integration weights
    megCoils->integrationWeights.resize( n_intpnt );
    megCoils->integrationWeights( 0 ) = 16.0 / 81.0;
    megCoils->integrationWeights( 1 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 2 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 3 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 4 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 5 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 6 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 7 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 8 ) = 10.0 / 81.0;
}

void WMegCoilInformation::neuromagCoil3024( WMegCoils* const megCoils )
{
    const PositionsT::Index n_intpnt = 9;
    // Integration points
    PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9;
    ip1 << 0.0, 0.0, 0.3;

    // +- 8.13, +- 8.13, 0.3mm
    ip2 << 8.13, 8.13, 0.3;
    ip3 << 8.13, -8.13, 0.3;
    ip4 << -8.13, 8.13, 0.3;
    ip5 << -8.13, -8.13, 0.3;

    //  0.0, +- 8.13, 0.3mm
    ip6 << 0.0, 8.13, 0.3;
    ip7 << 0.0, -8.13, 03;

    // +- 8.13, 0.0, 0.3mm
    ip8 << 8.13, 0.0, 0.3;
    ip9 << -8.13, 0.0, 0.3;

    megCoils->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    megCoils->integrationPoints.col( 0 ) = ip1;
    megCoils->integrationPoints.col( 1 ) = ip2;
    megCoils->integrationPoints.col( 2 ) = ip3;
    megCoils->integrationPoints.col( 3 ) = ip4;
    megCoils->integrationPoints.col( 4 ) = ip5;
    megCoils->integrationPoints.col( 5 ) = ip6;
    megCoils->integrationPoints.col( 6 ) = ip7;
    megCoils->integrationPoints.col( 7 ) = ip8;
    megCoils->integrationPoints.col( 8 ) = ip9;

    // Integration weights
    megCoils->integrationWeights.resize( n_intpnt );
    megCoils->integrationWeights( 0 ) = 16.0 / 81.0;
    megCoils->integrationWeights( 1 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 2 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 3 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 4 ) = 25.0 / 324.0;
    megCoils->integrationWeights( 5 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 6 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 7 ) = 10.0 / 81.0;
    megCoils->integrationWeights( 8 ) = 10.0 / 81.0;
}

