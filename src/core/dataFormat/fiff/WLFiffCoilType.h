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

#ifndef WLFIFFCOILTYPE_H_
#define WLFIFFCOILTYPE_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t coil_type_t;

    namespace CoilType
    {
        const coil_type_t NONE = 0;
        const coil_type_t EEG = 1;
        const coil_type_t EEG_BIPOLAR = 5;
        const coil_type_t VV_PLANAR_W = 3011;
        const coil_type_t VV_PLANAR_T1 = 3012;
        const coil_type_t VV_PLANAR_T2 = 3013;
        const coil_type_t VV_PLANAR_T3 = 3014;
        const coil_type_t VV_MAG_W = 3021;
        const coil_type_t VV_MAG_T1 = 3022;
        const coil_type_t VV_MAG_T2 = 3023;
        const coil_type_t VV_MAG_T3 = 3024;
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFCOILTYPE_H_
