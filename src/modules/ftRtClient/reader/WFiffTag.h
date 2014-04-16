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

#ifndef WFIFFTAG_H_
#define WFIFFTAG_H_

#include <fiff/fiff_tag.h>

using namespace FIFFLIB;

/**
 * Tags are used in front of data items to tell what they are.
 */
class WFiffTag: public FiffTag
{
public:

    /**
     * Read tag information of one tag from a fif file.
     * if pos is not provided, reading starts from the current file position
     *
     * @param[in] p_pStream opened fif file
     * @param[out] p_pTag the read tag info
     * @param[in] p_bDoSkip if true it skips the data of the tag (optional, default = true)
     *
     * @return true if succeeded, false otherwise
     */
    static bool read_tag_info( FiffStream* p_pStream, FiffTag::SPtr &p_pTag, bool p_bDoSkip = true );

    /**
     * Read one tag from a fif file.
     * if pos is not provided, reading starts from the current file position
     *
     * @param[in] p_pStream opened fif file
     * @param[out] p_pTag the read tag
     * @param[in] pos position of the tag inside the fif file
     *
     * @return true if succeeded, false otherwise
     */
    static bool read_tag( FiffStream* p_pStream, FiffTag::SPtr& p_pTag, qint64 pos = -1 );
};

#endif /* WFIFFTAG_H_ */
