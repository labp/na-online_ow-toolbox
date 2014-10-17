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

#ifndef WFIFFTAG_H_
#define WFIFFTAG_H_

#include <fiff/fiff_tag.h>

#include "WFiffStream.h"

/**
 * Tags are used in front of data items to tell what they are.
 */
class WFiffTag: public FIFFLIB::FiffTag
{
public:
    /**
     * Read tag information of one tag from a fif file.
     * if pos is not provided, reading starts from the current file position
     *
     * \param[out] p_pTag the read tag info
     * \param[in] p_pStream opened fif file
     * \param[in] p_bDoSkip if true it skips the data of the tag (optional, default = true)
     *
     * \return true if succeeded, false otherwise
     */
    static bool read_tag_info( FIFFLIB::FiffTag* const p_pTag, FIFFLIB::FiffStream* const p_pStream, bool p_bDoSkip = true );

    /**
     * Read one tag from a fif file.
     * if pos is not provided, reading starts from the current file position
     *
     * \param[out] p_pTag the read tag
     * \param[in] p_pStream opened fif file
     * \param[in] pos position of the tag inside the fif file
     *
     * \return true if succeeded, false otherwise
     */
    static bool read_tag( FIFFLIB::FiffTag* const p_pTag, FIFFLIB::FiffStream* const p_pStream, qint64 pos = -1 );
};

#endif  // WFIFFTAG_H_
