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

#ifndef WLFIFFLIB_H_
#define WLFIFFLIB_H_

#include <boost/cstdint.hpp>

namespace WLFiffLib
{
    typedef int16_t int16;
    typedef uint16_t uint16;

    typedef int32_t int32;
    typedef uint32_t uint32;

    typedef uint64_t uint64;

    typedef int32 enum_t;
    typedef float freq_t;
    typedef int32 ichan_t;
    typedef int32 nchan_t;
    typedef int32 icoil_t;
    typedef int32 ncoil_t;
    typedef int32 isamples_t;
    typedef int32 nsamples_t;
    typedef float time_t;
    typedef int32 ident_t;
    typedef int32 kind_t;
} /* namespace WLFiffLib */
#endif  // WLFIFFLIB_H_
