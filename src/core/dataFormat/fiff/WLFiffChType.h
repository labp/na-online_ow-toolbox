//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WLFIFFCHTYPE_H_
#define WLFIFFCHTYPE_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t ch_type_t;

    namespace ChType
    {
        const ch_type_t MAGN = 1;
        const ch_type_t EL = 2;
        const ch_type_t STIM = 3;
        const ch_type_t BIO = 102;
        const ch_type_t MCG = 201;
        const ch_type_t EOG = 202;
        const ch_type_t MAGN_REF = 301;
        const ch_type_t EMG = 302;
        const ch_type_t ECG = 402;
        const ch_type_t MISC = 502;
        const ch_type_t RESP = 602;
        const ch_type_t QUAT0 = 700;
        const ch_type_t QUAT1 = 701;
        const ch_type_t QUAT2 = 702;
        const ch_type_t QUAT3 = 703;
        const ch_type_t QUAT4 = 704;
        const ch_type_t QUAT5 = 705;
        const ch_type_t QUAT6 = 706;
        const ch_type_t HPI_GOODNESS = 707;
        const ch_type_t HPI_ERROR = 708;
        const ch_type_t HPI_MOVEMENT = 709;
        const ch_type_t SYST = 900;
        const ch_type_t IAS = 910;
        const ch_type_t EXCI = 920;
        const ch_type_t DIPOLE_WAVE = 1000;
        const ch_type_t GOODNESS_FIT = 1001;
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFCHTYPE_H_
