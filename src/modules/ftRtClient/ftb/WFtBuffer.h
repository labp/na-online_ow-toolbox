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

#ifndef WFTBUFFER_H_
#define WFTBUFFER_H_

#include <stdint.h>

#include <message.h>

namespace wftb
{
    typedef uint16_t version_t;
    typedef uint32_t bufsize_t;

    typedef messagedef_t MessageDefT;
    typedef message_t MessageT;

    typedef uint32_t nchans_t;
    typedef uint32_t isample_t;
    typedef uint32_t nsamples_t;
    typedef uint32_t nelements_t;

    typedef float fsamp_t;
    typedef uint32_t time_t;

    typedef datasel_t DataSelT;
    typedef headerdef_t HeaderDefT;
    typedef header_t HeaderT;
    typedef samples_events_t SamplesEventsT;
    typedef waitdef_t WaitDefT;
} /* namespace wftb */

#endif  // WFTBUFFER_H_
