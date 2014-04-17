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

#ifndef WFTCHUNKPROCESSOR_H_
#define WFTCHUNKPROCESSOR_H_

#include <boost/shared_ptr.hpp>

#include "core/container/WLArrayList.h"

#include "modules/ftRtClient/fieldtrip/dataTypes/WFTChunk.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/WFTChunkList.h"
#include "modules/ftRtClient/fieldtrip/processing/WFTIChunkProcessor.h"

/**
 * The WFTChunkProcessor class is created to apply some chunk specific operations on the FieldTrip header chunks managed in the WFTHeader class.
 * This is the right place to extend the implementation by chunk operations through inherit the WFTChunkProcessor.
 */
class WFTChunkProcessor: public WFTIChunkProcessor
{
public:

    /**
     * A shared pointer on a WFTChunkProcessor.
     */
    typedef boost::shared_ptr< WFTChunkProcessor > SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Destroys the WFTChunkProcessor.
     */
    virtual ~WFTChunkProcessor();

    /**
     * This method picks out the names of channels contained in the chunk list. Inside of a channel names chunk the values are represented as a '\0'
     * separated string. This implementation leaves the possibility open that the channel names comes in different chunks separately.
     *
     * @param chunkList A pointer on the list.
     * @return Returns a vector containing the found channel names.
     */
    virtual WLArrayList< std::string >::SPtr extractChannelNames( WFTChunkList::SPtr chunkList );

    /**
     * This method picks out the channel order and channel types included in the Channel Flags chunk.
     *
     * @param chunkList A list of header chunks.
     * @return Returns a list of strings.
     */
    //virtual WLArrayList< std::string >::SPtr extractChannelFlags( WFTChunkList::SPtr chunkList );
    /**
     * Processes a FieldTrip chunk containing a Neuromag header file to create the measurement information.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @param chunk The Neuromag header chunk.
     * @return Returns true if the header file could processed, otherwise false.
     */
    bool processNeuromagHeader( WFTChunk::SPtr chunk );

    /**
     * Gets the existing measurement information object.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @return The measurement information.
     */
    WFTHeader::MeasurementInfo_SPtr getMeasurementInfo();

    /**
     * Determines whether a measurement information object exists.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @return Returns true if there is a measurement information, otherwise false.
     */
    bool hasMeasurementInfo();

    /**
     * Extracts the channel names from the FieldTrip header chunk.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @param chunk The FieldTrip channel names chunk.
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    bool channelNamesChunk( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names );

    /**
     * Gets the channel names from a processed Neuromag header chunk. If there was no chunk yet, the method returns false.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    bool channelNamesMeasInfo( WLArrayList< std::string >::SPtr& names );

    /**
     * Gets the channel names form the Neuromag measurement information provided in the @chunk. The method overrides the existing
     * measurement information object.
     *
     * Inherited method from WFTIChunkProcessor.
     *
     * @param chunk The chunk containing the Neuromag header file.
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    bool channelNamesMeasInfo( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names );

protected:

    /**
     * The path to a temporary directory. It is platform dependent.
     */
    static const std::string TMPDIRPATH;

    /**
     * The name of the temporary Neuromag header FIFF file.
     */
    static const std::string TMPFILENAME;

    /**
     * The measurement information.
     */
    WFTHeader::MeasurementInfo_SPtr m_measInfo;

private:

    /**
     * This method extracts the channel names from a single chunk.
     *
     * @param chunk A pointer on the chunk.
     * @param vector A reference on the resulting string vector.
     * @return Returns true if there was channel names in the chunk, else false.
     */
    //bool getChannelNames( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr &vector );
};

#endif /* WFTCHUNKPROCESSOR_H_ */
