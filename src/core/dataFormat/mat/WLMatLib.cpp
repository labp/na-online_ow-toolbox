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

#include "WLMatLib.h"

bool WLMatLib::ArrayFlags::isComplex( const mArrayFlags_t& data )
{
    return data & ArrayFlags::MASK_COMPLEX;
}

bool WLMatLib::ArrayFlags::isGlobal( const mArrayFlags_t& data )
{
    return data & ArrayFlags::MASK_GLOBAL;
}

bool WLMatLib::ArrayFlags::isLogical( const mArrayFlags_t& data )
{
    return data & ArrayFlags::MASK_LOGICAL;
}

WLMatLib::mArrayType_t WLMatLib::ArrayFlags::getArrayType( const mArrayFlags_t& data )
{
    const mArrayFlags_t tmp = data & ArrayFlags::MASK_GET_CLASS;
    return tmp & ArrayFlags::MASK_GET_CLASS;
}

bool WLMatLib::ArrayTypes::isNumericArray( const mArrayType_t& type )
{
    if( type == ArrayTypes::mxDOUBLE_CLASS || type == ArrayTypes::mxSINGLE_CLASS )
    {
        return true;
    }
    if( type == ArrayTypes::mxINT8_CLASS || type == ArrayTypes::mxUINT8_CLASS )
    {
        return true;
    }
    if( type == ArrayTypes::mxINT16_CLASS || type == ArrayTypes::mxUINT16_CLASS )
    {
        return true;
    }
    if( type == ArrayTypes::mxINT32_CLASS || type == ArrayTypes::mxUINT32_CLASS )
    {
        return true;
    }
    return false;
}
