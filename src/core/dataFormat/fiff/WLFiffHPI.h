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

#ifndef CORE_DATAFORMAT_FIFF_WLFIFFHPI_H_
#define CORE_DATAFORMAT_FIFF_WLFIFFHPI_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    /**
     * Head position indicator (HPI) data.
     * \note These information are incomplete.
     *
     * \author pieloth
     * \ingroup fiff
     */
    namespace HPI
    {
        const tag_id_t NCOIL = 216; /**< Number of HPI coils. */
        const tag_id_t COIL_FREQ = 236; /**< HPI coil excitation frequency. */
    }
} /* namespace WLFiffLib */
#endif  // CORE_DATAFORMAT_FIFF_WLFIFFHPI_H_
