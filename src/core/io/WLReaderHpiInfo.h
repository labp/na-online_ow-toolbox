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

#ifndef WLREADERHPIINFO_H_
#define WLREADERHPIINFO_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <fiff/fiff_dir_tree.h>
#include <fiff/fiff_stream.h>

#include "core/data/WLEMMHpiInfo.h"

#include "WLReaderGeneric.h"

/**
 * Reads HPI information from a FIF file.
 *
 * \author pieloth
 * \ingroup io
 */
class WLReaderHpiInfo: public WLReaderGeneric< WLEMMHpiInfo >
{
public:
    static const std::string CLASS;

    /**
     * Shared pointer abbreviation to a instance of this class.
     */
    typedef boost::shared_ptr< WLReaderHpiInfo > SPtr;

    explicit WLReaderHpiInfo( std::string fname ) throw( WDHNoSuchFile );
    virtual ~WLReaderHpiInfo();

    virtual WLIOStatus::IOStatusT read( WLEMMHpiInfo* const hpiInfo );

private:
    bool readHpiMeas( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream, const FIFFLIB::FiffDirTree& tree );
    bool readHpiCoil( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream, const FIFFLIB::FiffDirTree& tree );
    bool readHpiResult( WLEMMHpiInfo* const hpiInfo, FIFFLIB::FiffStream* const stream, const FIFFLIB::FiffDirTree& tree );
};

#endif  // WLREADERHPIINFO_H_
