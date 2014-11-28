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

#ifndef WLFIFFUNIT_H_
#define WLFIFFUNIT_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t unit_t;

    /**
     * Value units.
     *
     * \author pieloth
     * \ingroup fiff
     */
    namespace Unit
    {
        const unit_t NONE = -1; /**< No unit */
        const unit_t UNITLESS = 0; /**< unitless */
        const unit_t M = 1; /**< meter */
        const unit_t KG = 2; /**< kilogram */
        const unit_t SEC = 3; /**< second */
        const unit_t A = 4; /**< ampere */
        const unit_t K = 5; /**< Kelvin */
        const unit_t MOL = 6; /**< mole */
        const unit_t RAD = 7; /**< radian */
        const unit_t SR = 8; /**< steradian */
        const unit_t CD = 9; /**< candela */
        const unit_t HZ = 101; /**< herz */
        const unit_t N = 102; /**< Newton */
        const unit_t PA = 103; /**< pascal */
        const unit_t J = 104; /**< joule */
        const unit_t W = 105; /**< watt */
        const unit_t C = 106; /**< coulomb */
        const unit_t V = 107; /**< volt */
        const unit_t F = 108; /**< farad */
        const unit_t OHM = 109; /**< ohm */
        const unit_t MHO = 110; /**< one per ohm */
        const unit_t WB = 111; /**< weber */
        const unit_t T = 112; /**< tesla */
        const unit_t H = 113; /**< Henry */
        const unit_t CEL = 114; /**< celcius */
        const unit_t LM = 115; /**< lumen */
        const unit_t LX = 116; /**< lux */
        const unit_t TM = 201; /**< T/m */
        const unit_t AM = 202; /**< Am */
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFUNIT_H_
