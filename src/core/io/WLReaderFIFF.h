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

#ifndef WLREADERFIFF_H_
#define WLREADERFIFF_H_

#include <string>

#include <core/dataHandler/io/WReader.h>

#include <libfiffio/common/LFReturnCodes.h>
#include <libfiffio/common/LFUnits.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMEnumTypes.h"

#include "core/io/WLReader.h"

namespace LaBP
{

    class WLReaderFIFF: public WLReader
    {
    public:

        static const string CLASS;

        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WLReaderFIFF( std::string fname );
        /**
         * Read the file and create a dataset out of it.
         */
        ReturnCode::Enum Read( WLEMMeasurement::SPtr out );
        /**
         * Reads subject data only.
         */
        ReturnCode::Enum Read( LaBP::WLEMMSubject::SPtr out );

    private:
        static ReturnCode::Enum getReturnCode( returncode_t rc );

        static LaBP::WEUnit::Enum getChanUnit( fiffunits_t unit );
    };
}
#endif /* WLREADERFIFF_H_ */
