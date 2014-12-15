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

#ifndef WLFIFFBEMSURFTYPE_H_
#define WLFIFFBEMSURFTYPE_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t bem_surf_type_t;

    /**
     * Surface identifiers.
     *
     * \author pieloth
     * \ingroup fiff
     */
    namespace BEMSurfType
    {
        const bem_surf_type_t UNKNOWN2 = -1;
        const bem_surf_type_t UNKNOWN = 0;
        const bem_surf_type_t BRAIN = 1;
        const bem_surf_type_t CSF = 2;
        const bem_surf_type_t SKULL = 3;
        const bem_surf_type_t HEAD = 4;
        const bem_surf_type_t BLOOD = 11;
        const bem_surf_type_t HEART = 12;
        const bem_surf_type_t LUNGS = 13;
        const bem_surf_type_t TORSO = 14;
        const bem_surf_type_t NM122 = 22;
        const bem_surf_type_t UNIT_SPHERE = 22;
        const bem_surf_type_t VV = 23;
    }
} /* namespace WLFiffLib */

#endif  // WLFIFFBEMSURFTYPE_H_
