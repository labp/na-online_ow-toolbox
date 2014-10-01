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

#ifndef WLFIFFCOORDSYSTEM_H_
#define WLFIFFCOORDSYSTEM_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t coord_system_t;

    namespace CoordSystem
    {
        const coord_system_t UNKNOWN = 0;
        const coord_system_t DEVICE = 1;
        const coord_system_t ISOTRAK = 2;
        const coord_system_t HPI = 3;
        const coord_system_t HEAD = 4;
        const coord_system_t DATA_VOLUME = 5;
        const coord_system_t DATA_SLICE = 6;
        const coord_system_t DATA_DISPLAY = 7;
        const coord_system_t DICOM_DEVICE = 8;
        const coord_system_t IMAGING_DEVICE = 9;
        const coord_system_t VOXEL_DATA = 10;
        const coord_system_t ATLAS_HEAD = 11;
        const coord_system_t TORSO = 100;
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFCOORDSYSTEM_H_
