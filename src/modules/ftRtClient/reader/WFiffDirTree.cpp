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

#include "WFiffDirTree.h"

bool WFiffDirTree::find_tag( FiffTag* const p_pTag, FiffStream* const p_pStream, fiff_int_t findkind ) const
{
    if( p_pTag == NULL )
    {
        return false;
    }

    for( qint32 p = 0; p < this->nent; ++p )
    {
        if( this->dir[p].kind == findkind )
        {
            WFiffTag::read_tag( p_pTag, p_pStream, this->dir[p].pos );
            return true;
        }
    }

    return false;
}
