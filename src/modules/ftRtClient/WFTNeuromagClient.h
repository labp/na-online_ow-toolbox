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

#ifndef WFTNEUROMAGCLIENT_H
#define WFTNEUROMAGCLIENT_H

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMDRaw.h"
#include "core/data/WLEMMeasurement.h"

#include "ftb/WFtbChunk.h"
#include "ftbClient/WFTRtClient.h"

/**
 * WFTNeuromagClient represents the basic streaming client class for the Elekta/Neuromag device. It inherits the WFTRtClient class and extends them
 * for handling the data as WLEMMeasurement objects used in OpenWalnut.
 *
 * \author maschke
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
     * \return Returns true if the client is streaming, else false.
     */
    bool isStreaming() const;

    /**
     * This method represents the signal to the client to preparing for streaming.
     * Doing this the client tries to receive the header structure from the server. If the acquisition software is not running the start
     * will fail and no header information can be retrieved.
     *
     * \return Returns true if the preparation was successful, else false.
     */
    bool start();

    /**
     * Method to switch the client into state "not streaming".
     */
    void stop();

    /**
     * After getting data from the server this method can be used to create a WLEMMeasurement object for the process chain.
     *
     * \param emm The WLEMMeasurement object to fill.
     * \return Returns false in case of problems occur during EMM creation, else true.
     */
    bool createEMM( WLEMMeasurement::SPtr emm );

    /**
     * Gets whether or not the client applies scaling factors on the samples.
     *
     * \return Returns true if the client applies the scaling factors, otherwise false.
     */
    bool isScalingApplied() const;

    /**
     * Set whether or not the client has to apply scaling factors on the samples.
     *
     * \param applyScaling The flag.
     */
    void setScaling( bool applyScaling );

protected:
    /**
     * Creates a raw EMM object with all modalities in one data matrix.
     *
     * \param emm The EMM object.
     * \return Returns false in case of problems occur during EMM creation, otherwise true.
     */
    bool getRawData( WLEMDRaw::SPtr* const rawData );

private:
    /**
     * Preparation method for starting the client streaming.
     *
     * \return Returns true if the header can be received, else false.
     */
    bool prepareStreaming();

    /**
     * Flag to define whether the client is running.
     */
    bool m_streaming;

    /**
     * Flag to determine whether the scaling factors have to apply on the samples.
     */
    bool m_applyScaling;
};

#endif  // WFTNEUROMAGCLIENT_H
