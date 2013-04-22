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

#ifndef WREADERFIFF_H_
#define WREADERFIFF_H_

#include <string>

#include <core/dataHandler/io/WReader.h>

#include <libfiffio/common/LFReturnCodes.h>
#include <libfiffio/common/LFUnits.h>

#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

namespace LaBP
{

    class WReaderFIFF: public WReader
    {
    public:

        static const string CLASS;

        enum ReturnCode
        {
            SUCCESS, /**< Normal */
            ERROR_FOPEN, /**< Error opening file */
            ERROR_FREAD, /**< File read error */
            ERROR_UNKNOWN /**< Unknown error */
        };

        /**
         * Constructs a reader object.
         *
         * \param fname path to file which should be loaded
         */
        explicit WReaderFIFF( std::string fname );
        /**
         * Read the file and create a dataset out of it.
         */
        ReturnCode Read( LaBP::WDataSetEMM::SPtr out );
        /**
         * Reads subject data only.
         */
        ReturnCode Read( LaBP::WDataSetEMMSubject::SPtr out );
        /**
         * Reads raw data only.
         */
        ReturnCode Read( std::vector< std::vector< double > >& out );

    private:
        static ReturnCode getReturnCode( returncode_t rc );

        static LaBP::WEUnit::Enum getChanUnit( fiffunits_t unit );
    };
}
#endif /* WREADERFIFF_H_ */
