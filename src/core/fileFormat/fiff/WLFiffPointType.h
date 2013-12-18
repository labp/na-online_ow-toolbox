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

#ifndef WLFIFFPOINTTYPE_H_
#define WLFIFFPOINTTYPE_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    typedef enum_t point_type_t;

    namespace PointType
    {
        const point_type_t CARDINAL = 1;
        const point_type_t HPI = 2;
        const point_type_t EEG = 3;
        const point_type_t ECG = 3;
        const point_type_t EXTRA = 4;
        const point_type_t HEAD_SURFACE = 5;
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFPOINTTYPE_H_
