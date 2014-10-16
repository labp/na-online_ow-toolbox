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

#ifndef WFIFFDIRTREE_H_
#define WFIFFDIRTREE_H_

#include <fiff/fiff_dir_tree.h>

#include "WFiffTag.h"

using namespace FIFFLIB;

/**
* Replaces _fiffDirNode struct
*
* @brief Directory tree structure
*/
class WFiffDirTree: public FiffDirTree
{
public:

    /**
     * Implementation of the find_tag function in various files e.g. fiff_read_named_matrix.m
     *
     * Founds a tag of a given kind within a tree, and reeds it from file.
     * Note: In difference to mne-matlab this is not a static function. This is a method of the WFiffDirTree
     *       class, that's why a tree object doesn't need to be handed to the function.
     *
     * @param[in] p_pStream the opened fif file
     * @param[in] findkind the kind which should be found
     * @param[out] p_pTag the found tag
     *
     * @return true if found, false otherwise
     */
    bool find_tag( FiffStream* p_pStream, fiff_int_t findkind, FiffTag::SPtr& p_pTag ) const;

};

#endif /* WFIFFDIRTREE_H_ */
