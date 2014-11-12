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

#ifndef WFTCHUNKREADER_H_
#define WFTCHUNKREADER_H_

#include <map>

#include <boost/shared_ptr.hpp>

#include "../dataTypes/WFTChunk.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDRaw.h"

/**
 * The interface provides access to extended header information a.k.a. chunks.
 *
 * \author pieloth
 */
class WFTChunkReader
{
public:
    typedef boost::shared_ptr< WFTChunkReader > SPtr;
    typedef boost::shared_ptr< const WFTChunkReader > ConstSPtr;

    typedef std::map< wftb::chunk_type_t, WFTChunkReader::SPtr > MapT;

    virtual ~WFTChunkReader();

    /**
     * Returns the supported chunk type.
     *
     * \return supported chunk type
     */
    virtual wftb::chunk_type_t supportedChunkType() const = 0;

    /**
     * Reads/processes/parses a chunk and stores the data.
     *
     * \param chunk Chunk to read.
     * \return False, if no data was read.
     */
    virtual bool read( WFTChunk::ConstSPtr chunk ) = 0;

    /**
     * Applies/sets the read data to an EMMeasurement object.
     *
     * \param emm Instance which should hold the data.
     * \param raw Raw data e.g. EEG, MEG.
     * \return False, if no data was applied/set.
     */
    virtual bool apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw ) = 0;
};

#endif  // WFTCHUNKREADER_H_
