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

#ifndef WFTCLIENTSTREAMING_H_
#define WFTCLIENTSTREAMING_H_

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMDRaw.h"
#include "core/data/WLEMMeasurement.h"

#include "fieldtrip/client/WFTRtClient.h"

/**
 * WFTNeuromagClient represents the basic streaming client class for the Elekta/Neuromag device. It inherits the WFTRtClient class and extends them
 * for handling the data as WLEMMeasurement objects used in Openwalnut.
 */
class WFTNeuromagClient: public WFTRtClient
{
public:

    /**
     * Shared pointer on a WFTNeuromagClient.
     */
    typedef boost::shared_ptr< WFTNeuromagClient > SPtr;

    /**
     * Contains the class name.
     */
    static const std::string CLASS;

    /**
     * Constructs the WFTNeuromagClient object.
     */
    WFTNeuromagClient();

    /**
     * Gets whether the client is streaming. Is it so, the client has an open connection to the FieldTrip buffer server.
     *
     * @return Returns true if the client is streaming, else false.
     */
    bool isStreaming() const;

    /**
     * This method represents the signal to the client to preparing for streaming.
     * Doing this the client tries to receive the header structure from the server. If the acquisition software is not running the start
     * will fail and no header information can be retrieved.
     *
     * @return Returns true if the preparation was successful, else false.
     */
    bool start();

    /**
     * Method to switch the client into state "not streaming".
     */
    void stop();

    /**
     * After getting data from the server this method can be used to create a WLEMMeasurement object for the process chain.
     *
     * @param emm The WLEMMeasurement object to fill.
     * @return Returns false in case of problems occur during EMM creation, else true.
     */
    bool createEMM( WLEMMeasurement::SPtr emm );

protected:

    /**
     * Creates a raw EMM object with all modalities in one data matrix.
     *
     * @param emm The EMM object.
     * @return Returns false in case of problems occur during EMM creation, otherwise true.
     */
    bool getRawData( WLEMDRaw::SPtr& modality );

    /**
     * Creates a EMM object with a more detailed appearance. The modalities are splitted in several modalities and the EMM
     * contains some measurement information.
     *
     * @param emm The EMM object.
     * @return Returns false in case of problems occur during EMM creation, otherwise true.
     */
    bool createDetailedEMM( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr rawData );

private:

    /**
     * Preparation method for starting the client streaming.
     *
     * @return Returns true if the header can be received, else false.
     */
    bool prepareStreaming();

    /**
     * Flag to define whether the client is running.
     */
    bool m_streaming;

};

#endif /* WFTCLIENTSTREAMING_H_ */
