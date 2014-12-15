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

#ifndef WLFIFFUNITMULTIPLIER_H_
#define WLFIFFUNITMULTIPLIER_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t unitm_t;

    /**
     * Value multipliers.
     *
     * \author pieloth
     * \ingroup fiff
     */
    namespace UnitMultiplier
    {
        const unitm_t ET = 18; /**< 10^18 */
        const unitm_t PET = 15; /**< 10^15 */
        const unitm_t T = 12; /**< 10^12 */
        const unitm_t GIG = 9; /**< 10^9 */
        const unitm_t MEG = 6; /**< 10^6 */
        const unitm_t K = 3; /**< 10^3 */
        const unitm_t H = 2; /**< 10^2 */
        const unitm_t DA = 1; /**< 10^1 */
        const unitm_t NONE = 0; /**< 10^0 */
        const unitm_t D = -1; /**< 10^-1 */
        const unitm_t C = -2; /**< 10^-2 */
        const unitm_t M = -3; /**< 10^-3 */
        const unitm_t MU = -6; /**< 10^-6 */
        const unitm_t N = -9; /**< 10^-9 */
        const unitm_t P = -12; /**< 10^-12 */
        const unitm_t F = -15; /**< 10^-15 */
        const unitm_t A = -18; /**< 10^-18 */
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFUNITMULTIPLIER_H_
