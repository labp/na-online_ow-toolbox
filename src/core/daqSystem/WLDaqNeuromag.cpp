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

#include "core/data/WLMegCoilInfo.h"
#include "core/data/enum/WLEExponent.h"

#include "WLDaqNeuromag.h"

static const double MM_TO_M = WLEExponent::factor( WLEExponent::MILLI );

void WLDaqNeuromag::applyIntegrationPoints3012( WLMegCoilInfo* const megCoil )
{
    // TODO(pieloth): Check assumption that we map 1 gradiometer to 2 magnetometer.
    // map 1 gradiometer to "2 magnetometer" by just using the integration points and different weights (+/-).
    const WLMegCoilInfo::PositionsT::Index n_intpnt = 16;

    // Integration points
    megCoil->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    WLMegCoilInfo::PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8;

    // Integration weights
    // +- 1 / ( 4*16.2mm )
    megCoil->integrationWeights.resize( n_intpnt );
    const double iw1 = 1.0 / ( 4 * 16.2 * MM_TO_M ); // TODO(pieloth): verify mm to m
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

    megCoil->integrationPoints.col( 0 ) = ip1 * MM_TO_M;
    megCoil->integrationPoints.col( 1 ) = ip2 * MM_TO_M;
    megCoil->integrationPoints.col( 2 ) = ip3 * MM_TO_M;
    megCoil->integrationPoints.col( 3 ) = ip4 * MM_TO_M;
    megCoil->integrationPoints.col( 4 ) = ip5 * MM_TO_M;
    megCoil->integrationPoints.col( 5 ) = ip6 * MM_TO_M;
    megCoil->integrationPoints.col( 6 ) = ip7 * MM_TO_M;
    megCoil->integrationPoints.col( 7 ) = ip8 * MM_TO_M;

    megCoil->integrationWeights( 0 ) = iw1;
    megCoil->integrationWeights( 1 ) = iw1;
    megCoil->integrationWeights( 2 ) = iw1;
    megCoil->integrationWeights( 3 ) = iw1;
    megCoil->integrationWeights( 4 ) = iw1;
    megCoil->integrationWeights( 5 ) = iw1;
    megCoil->integrationWeights( 6 ) = iw1;
    megCoil->integrationWeights( 7 ) = iw1;

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

    megCoil->integrationPoints.col( 8 ) = ip1 * MM_TO_M;
    megCoil->integrationPoints.col( 9 ) = ip2 * MM_TO_M;
    megCoil->integrationPoints.col( 10 ) = ip3 * MM_TO_M;
    megCoil->integrationPoints.col( 11 ) = ip4 * MM_TO_M;
    megCoil->integrationPoints.col( 12 ) = ip5 * MM_TO_M;
    megCoil->integrationPoints.col( 13 ) = ip6 * MM_TO_M;
    megCoil->integrationPoints.col( 14 ) = ip7 * MM_TO_M;
    megCoil->integrationPoints.col( 15 ) = ip8 * MM_TO_M;

    megCoil->integrationWeights( 8 ) = iw2;
    megCoil->integrationWeights( 9 ) = iw2;
    megCoil->integrationWeights( 10 ) = iw2;
    megCoil->integrationWeights( 11 ) = iw2;
    megCoil->integrationWeights( 12 ) = iw2;
    megCoil->integrationWeights( 13 ) = iw2;
    megCoil->integrationWeights( 14 ) = iw2;
    megCoil->integrationWeights( 15 ) = iw2;
}

void WLDaqNeuromag::applyIntegrationPoints3014( WLMegCoilInfo* const megCoil )
{
    // 3012 and 3014 has the same geometry
    applyIntegrationPoints3012( megCoil );
}

void WLDaqNeuromag::applyIntegrationPoints3022( WLMegCoilInfo* const megCoil )
{
    const WLMegCoilInfo::PositionsT::Index n_intpnt = 9;
    // Integration points
    WLMegCoilInfo::PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9;
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

    megCoil->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    megCoil->integrationPoints.col( 0 ) = ip1 * MM_TO_M;
    megCoil->integrationPoints.col( 1 ) = ip2 * MM_TO_M;
    megCoil->integrationPoints.col( 2 ) = ip3 * MM_TO_M;
    megCoil->integrationPoints.col( 3 ) = ip4 * MM_TO_M;
    megCoil->integrationPoints.col( 4 ) = ip5 * MM_TO_M;
    megCoil->integrationPoints.col( 5 ) = ip6 * MM_TO_M;
    megCoil->integrationPoints.col( 6 ) = ip7 * MM_TO_M;
    megCoil->integrationPoints.col( 7 ) = ip8 * MM_TO_M;
    megCoil->integrationPoints.col( 8 ) = ip9 * MM_TO_M;

    // Integration weights
    megCoil->integrationWeights.resize( n_intpnt );
    megCoil->integrationWeights( 0 ) = 16.0 / 81.0;
    megCoil->integrationWeights( 1 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 2 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 3 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 4 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 5 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 6 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 7 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 8 ) = 10.0 / 81.0;
}

void WLDaqNeuromag::applyIntegrationPoints3024( WLMegCoilInfo* const megCoil )
{
    const WLMegCoilInfo::PositionsT::Index n_intpnt = 9;
    // Integration points
    WLMegCoilInfo::PositionT ip1, ip2, ip3, ip4, ip5, ip6, ip7, ip8, ip9;
    ip1 << 0.0, 0.0, 0.3;

    // +- 8.13, +- 8.13, 0.3mm
    ip2 << 8.13, 8.13, 0.3;
    ip3 << 8.13, -8.13, 0.3;
    ip4 << -8.13, 8.13, 0.3;
    ip5 << -8.13, -8.13, 0.3;

    //  0.0, +- 8.13, 0.3mm
    ip6 << 0.0, 8.13, 0.3;
    ip7 << 0.0, -8.13, 0.3;

    // +- 8.13, 0.0, 0.3mm
    ip8 << 8.13, 0.0, 0.3;
    ip9 << -8.13, 0.0, 0.3;

    megCoil->integrationPoints.resize( Eigen::NoChange, n_intpnt );
    megCoil->integrationPoints.col( 0 ) = ip1 * MM_TO_M;
    megCoil->integrationPoints.col( 1 ) = ip2 * MM_TO_M;
    megCoil->integrationPoints.col( 2 ) = ip3 * MM_TO_M;
    megCoil->integrationPoints.col( 3 ) = ip4 * MM_TO_M;
    megCoil->integrationPoints.col( 4 ) = ip5 * MM_TO_M;
    megCoil->integrationPoints.col( 5 ) = ip6 * MM_TO_M;
    megCoil->integrationPoints.col( 6 ) = ip7 * MM_TO_M;
    megCoil->integrationPoints.col( 7 ) = ip8 * MM_TO_M;
    megCoil->integrationPoints.col( 8 ) = ip9 * MM_TO_M;

    // Integration weights
    megCoil->integrationWeights.resize( n_intpnt );
    megCoil->integrationWeights( 0 ) = 16.0 / 81.0;
    megCoil->integrationWeights( 1 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 2 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 3 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 4 ) = 25.0 / 324.0;
    megCoil->integrationWeights( 5 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 6 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 7 ) = 10.0 / 81.0;
    megCoil->integrationWeights( 8 ) = 10.0 / 81.0;
}

