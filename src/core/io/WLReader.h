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

#ifndef WLREADER_H_
#define WLREADER_H_

#include <string>

#include <core/dataHandler/io/WReader.h>

class WLReader: public WReader
{
public:
    struct ReturnCode
    {
        enum Enum
        {
            SUCCESS, /**< Normal */
            ERROR_FOPEN, /**< Error opening file */
            ERROR_FREAD, /**< File read error */
            ERROR_UNKNOWN /**< Unknown error */
        };
    };

    explicit WLReader( std::string fname );
};

#endif  // WLREADER_H_
