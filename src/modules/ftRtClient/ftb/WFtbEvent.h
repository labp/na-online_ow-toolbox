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

#ifndef WFTBEVENT_H_
#define WFTBEVENT_H_

#include <message.h>

namespace wftb
{
    typedef eventdef_t EventDefT;
    typedef event_t EventT;

    typedef uint32_t ievent_t;
    typedef uint32_t nevents_t;

    namespace Event
    {
        typedef int32_t sample_t;
        typedef int32_t offset_t;
        typedef int32_t duration_t;
        typedef uint32_t type_type_t;
        typedef uint32_t value_type_t;
    } /* namespace Event */
} /* namespace wftb */
#endif  // WFTBEVENT_H_
