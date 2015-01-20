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

#ifndef WLDAQNEUROMAG_H_
#define WLDAQNEUROMAG_H_

#include <Eigen/Core>

#include "core/data/WLMegCoilInfo.h"
//class WLMegCoilInfo;

/**
 * Information about MEG coil geometry.
 * \see \cite SrcModelling2008
 *
 * \author pieloth
 */
namespace WLDaqNeuromag
{
    /**
     * Accurate coil descriptions for type T1 planar gradiometer, 3012.
     *
     * \param megCoil
     */
    void applyIntegrationPoints3012( WLMegCoilInfo* const megCoil );

    /**
     * Accurate coil descriptions for type T3 planar gradiometer, 3014.
     *
     * \param megCoil
     */
    void applyIntegrationPoints3014( WLMegCoilInfo* const megCoil );

    /**
     * Accurate coil descriptions for type T1 magnetometer, 3022.
     *
     * \param megCoil
     */
    void applyIntegrationPoints3022( WLMegCoilInfo* const megCoil );

    /**
     * Accurate coil descriptions for type T3 magnetometer, 3024.
     *
     * \param megCoil
     */
    void applyIntegrationPoints3024( WLMegCoilInfo* const megCoil );
}

#endif  // WLDAQNEUROMAG_H_
