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

#ifndef WFTCHUNKCHANNAMES_H_
#define WFTCHUNKCHANNAMES_H_

#include <list>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"

#include "WFTAChunk.h"

/**
 * The WFTChunkChanNames represents the FieldTrip header chunk, which contains the channel names.
 */
class WFTChunkChanNames: public WFTAChunk
{
public:
    /**
     * A shared pointer on a WFTChunkChanNames.
     */
    typedef boost::shared_ptr< WFTChunkChanNames > SPtr;

    /**
     * A shared pointer on a constant WFTChunkChanNames.
     */
    typedef boost::shared_ptr< const WFTChunkChanNames > ConstSPtr;

    /**
     * A channel names map with a modality type as key and a string list as value.
     */
    typedef std::map< WLEModality::Enum, WLArrayList< std::string >::SPtr > ChanNamesMapT;

    /**
     * A shared pointer on a channel names map.
     */
    typedef boost::shared_ptr< ChanNamesMapT > ChanNamesMap_SPtr;

    /**
     * A shared pointer on a constant channel names map.
     */
    typedef boost::shared_ptr< const ChanNamesMapT > ChanNamesMap_ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WFTChunkChanNames.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     */
    explicit WFTChunkChanNames( const char* data, const wftb::chunk_size_t size );

    /**
     * Gets the data as a smart storage structure. This method is used to serialize a chunk into a request message body.
     *
     * Inherited method from WFTAChunk.
     *
     * \return Returns a shared pointer on a constant smart storage.
     */
    WLSmartStorage::ConstSPtr serialize() const;

    /**
     * Gets the channel names for the @modality type.
     *
     * \return Returns a shared pointer on a constant channel names list.
     */
    WLArrayList< std::string >::ConstSPtr getData( const WLEModality::Enum modality ) const;

protected:
    /**
     * Based on the stored memory of @data, this method creates the chunks data structure.
     * It has to implement by a deriving class for a special chunk type.
     *
     * Inherited method from WFTAChunk.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     *
     * \return Returns true if the processing was successful, otherwise false.
     */
    bool process( const char* data, size_t size );

    /**
     * The channel names map.
     */
    ChanNamesMap_SPtr m_namesMap;

private:
    /**
     * A string - modality type map.
     */
    typedef std::map< std::string, WLEModality::Enum > ChanNameLabelT;

    /**
     * A static string - modality type map for a mapping during chunk processing between a modality and its string label inside of the chunk.
     */
    ChanNameLabelT m_modalityLabels;

    /**
     * Fills m_modalityLabels with its labels.
     */
    void insertLabels();

    /**
     * Splits the @str at the @delim's positions into the result list.
     *
     * \param result The result list.
     * \param str The string to split.
     * \param delim The separator character.
     */
    void split( std::list< std::string >* const result, const std::string& str, const char delim );
};

#endif  // WFTCHUNKCHANNAMES_H_
