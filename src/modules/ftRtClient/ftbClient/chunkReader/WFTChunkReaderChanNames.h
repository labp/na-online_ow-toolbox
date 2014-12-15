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

#ifndef WFTCHUNKREADERCHANNAMES_H_
#define WFTCHUNKREADERCHANNAMES_H_

#include <list>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"

#include "WFTChunkReader.h"

/**
 * Reads the header "FT_CHUNK_CHANNEL_NAMES", which contains the channel names.
 *
 * \authors maschke, pieloth
 */
class WFTChunkReaderChanNames: public WFTChunkReader
{
public:
    /**
     * A shared pointer on a WFTChunkReaderChanNames.
     */
    typedef boost::shared_ptr< WFTChunkReaderChanNames > SPtr;

    /**
     * A shared pointer on a constant WFTChunkReaderChanNames.
     */
    typedef boost::shared_ptr< const WFTChunkReaderChanNames > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    WFTChunkReaderChanNames();

    virtual ~WFTChunkReaderChanNames();

    virtual wftb::chunk_type_t supportedChunkType() const;

    virtual bool read( WFTChunk::ConstSPtr chunk );

    virtual bool apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw );

private:
    bool setChannelNames( WLEMMeasurement::SPtr emm );

    bool extractEmdsByNames( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw );

    bool extractEmdByName( std::string id, WLEMData::SPtr emd, WLEMDRaw::ConstSPtr raw );

    /**
     * A channel names map with a modality type as key and a string list as value.
     */
    typedef std::map< WLEModality::Enum, WLArrayList< std::string >::SPtr > ChanNamesMapT;

    /**
     * The channel names map.
     */
    ChanNamesMapT m_namesMap;

    std::list< std::string > m_namesAll;

    /**
     * A string - modality type map.
     */
    typedef std::map< std::string, WLEModality::Enum > ChanNameLabelT;

    /**
     * A static string - modality type map for a mapping during chunk processing between a modality and its string label inside of the chunk.
     */
    ChanNameLabelT m_modalityLabels;

    /**
     * Splits the @str at the @delim's positions into the result list.
     *
     * \param result The result list.
     * \param str The string to split.
     * \param delim The separator character.
     */
    void split( std::list< std::string >* const result, const std::string& str, const char delim );
};

#endif  // WFTCHUNKREADERCHANNAMES_H_
