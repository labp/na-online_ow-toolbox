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

#ifndef WFTCHANNELNAMES_H_
#define WFTCHANNELNAMES_H_

#include <map>

#include <boost/shared_ptr.hpp>

#include <fiff/fiff_info.h>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEModality.h"

#include "WFTChunk.h"

using namespace std;

/**
 *
 */
class WFTChannelNames: public WFTChunk
{

public:

    /**
     * A shared pointer on a channel names chunk.
     */
    typedef boost::shared_ptr< WFTChannelNames > SPtr;

    /**
     * A pointer on the measurement information.
     */
    typedef boost::shared_ptr< FIFFLIB::FiffInfo > MeasurementInfo_SPtr;

    /**
     * Constructs a new WFTChannelNames class.
     */
    WFTChannelNames();

    /**
     * Gets the channel names for the modality @type.
     *
     * @param type The modality type.
     * @return Returns the channel names.
     */
    WLArrayList< std::string >::SPtr getChannelNames( WLEModality::Enum type );

    /**
     * Gets all channel names.
     *
     * @return Returns a list including all channel names.
     */
    WLArrayList< std::string >::SPtr getChannelNames();

    /**
     * Creates the channel names from a Neuromag measurement information object.
     *
     * @param measInfo The measurement information object.
     * @return Returns true if the channel names were extracted, otherwise false.
     */
    bool fromFiff( MeasurementInfo_SPtr measInfo );


protected:

    /**
     * A channel names map with a modality type as key and a string list as value.
     */
    typedef map< WLEModality::Enum, WLArrayList< std::string >::SPtr > ChanNamesMapT;

    /**
     * A shared pointer on a channel names map.
     */
    typedef boost::shared_ptr< ChanNamesMapT > ChanNamesMapSPtr;

    /**
     * The channel names map.
     */
    ChanNamesMapSPtr m_namesMap;
};

#endif /* WFTCHANNELNAMES_H_ */
