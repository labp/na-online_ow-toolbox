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

#ifndef WLFIFFBLOCKTYPE_H_
#define WLFIFFBLOCKTYPE_H_

#include "WLFiffLib.h"

namespace WLFiffLib
{
    /**
     * These tags are used to divide a FIFF file into logical blocks.
     * \note This list is incomplete.
     *
     * \author pieloth
     * \ingroup fiff
     */
    namespace BlockType
    {
        const block_id_t ISOTRAK = 107; /**< Head digitization data. */
        const block_id_t HPI_MEAS = 108; /**< HPI measurement. */
        const block_id_t HPI_RESULT = 109; /**< Result of a HPI fitting procedure. */
        const block_id_t HPI_COIL = 110; /**< Data acquired from one HPI coil. */

        const block_id_t BEM = 310; /**< A boundary-element model (BEM) description. */
        const block_id_t BEM_SURF = 311; /**< Describes one BEM surface. */
    }
} /* namespace WLFiffLib */
#endif  // WLFIFFBLOCKTYPE_H_
