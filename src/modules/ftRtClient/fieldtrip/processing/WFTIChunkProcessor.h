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

#ifndef WFTICHUNKPROCESSOR_H_
#define WFTICHUNKPROCESSOR_H_

#include <boost/shared_ptr.hpp>

#include "core/container/WLArrayList.h"

#include "modules/ftRtClient/fieldtrip/dataTypes/chunks/WFTChunk.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/chunks/WFTChunkList.h"
#include "modules/ftRtClient/fieldtrip/dataTypes/WFTHeader.h"

/**
 * The WFTIChunkProcessor interface provides typical methods for processing FieldTrip header chunks.
 */
class WFTIChunkProcessor
{
public:

    /**
     * A shared pointer on a WFTIChunkProcessor.
     */
    typedef boost::shared_ptr< WFTIChunkProcessor > SPtr;

    /**
     * Destroys the WFTIChunkProcessor.
     */
    virtual ~WFTIChunkProcessor();

    /**
     * Processes a FieldTrip chunk containing a Neuromag header file to create the measurement information.
     *
     * @param chunk The Neuromag header chunk.
     * @return Returns true if the header file could processed, otherwise false.
     */
    virtual bool processNeuromagHeader( WFTChunk::SPtr chunk ) = 0;

    /**
     * Gets the existing measurement information object.
     *
     * @return The measurement information.
     */
    virtual WFTHeader::MeasurementInfo_SPtr getMeasurementInfo() = 0;

    /**
     * Determines whether a measurement information object exists.
     *
     * @return Returns true if there is a measurement information, otherwise false.
     */
    virtual bool hasMeasurementInfo() = 0;

    /**
     * Extracts the channel names from the FieldTrip header chunk.
     *
     * @param chunk The FieldTrip channel names chunk.
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    virtual bool channelNamesChunk( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names ) = 0;

    /**
     * Gets the channel names from a processed Neuromag header chunk. If there was no chunk yet, the method returns false.
     *
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    virtual bool channelNamesMeasInfo( WLArrayList< std::string >::SPtr& names ) = 0;

    /**
     * Gets the channel names form the Neuromag measurement information provided in the @chunk. The method overrides the existing
     * measurement information object.
     *
     * @param chunk The chunk containing the Neuromag header file.
     * @param names The channel names vector to fill.
     * @return Returns true if the channel names was extracted, otherwise false.
     */
    virtual bool channelNamesMeasInfo( WFTChunk::SPtr chunk, WLArrayList< std::string >::SPtr& names ) = 0;

    /**
     * Processes a FieldTrip chunk containing a Neuromag Isotrak file to extract the digitalization points.
     *
     * Inherited mehod from WFTIChunkProcessor.
     *
     * @param chunk The chunk.
     * @return Returns true if the Isotrak file could processed, otherwise false.
     */
    virtual bool processNeuromagIsotrak( WFTChunk::SPtr chunk ) = 0;

};

#endif /* WFTICHUNKPROCESSOR_H_ */
