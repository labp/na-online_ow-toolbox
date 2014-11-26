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

#ifndef WMFTRTCLIENT_H_
#define WMFTRTCLIENT_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMSubject.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "ftbClient/WFtbClient.h"
#include "ftbClient/network/WFTConnection.h"


/**
 * The FieldTrip Real-time Client implements a streaming client from a FieldTrip Buffer server used by several EEG/ MEG acquisition systems.
 * The client receives data from the buffer server and computes them into the internal data strctures. After that the data will be send into
 * the processing chain of the OpenWalnutToolbox.
 *
 * \author maschke
 * \ingroup io
 */
class WMFTRtClient: public WLModuleDrawable
{
public:
    /**
     * Constructs a new WMFTRtClient.
     */
    WMFTRtClient();

    /**
     * Destroys the WMFTRtClient.
     */
    virtual ~WMFTRtClient();

    /**
     * Gives back the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * Gives back a description of this module.
     * \return description to module.
     */
    virtual const std::string getDescription() const;

protected:
    /**
     * Method for initialize the module.
     *
     * Inherited method from WLModuleDrawable.
     */
    virtual void moduleInit();

    /**
     * Entry point after loading the module. Runs in separate thread.
     */
    virtual void moduleMain();

    /**
     * Initialize the connectors this module is using.
     */
    virtual void connectors();

    /**
     * Initialize the properties for this module.
     */
    virtual void properties();

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual boost::shared_ptr< WModule > factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

    /**
     * Inherited method from WLEMMCommandProcessor.
     *
     * \param emmIn The measurement object.
     * \return Returns true if the computaion was successfully, otherwise false.
     */
    virtual bool processCompute( WLEMMeasurement::SPtr emmIn );

    /**
     * Inherited method from WLEMMCommandProcessor.
     *
     * \param cmd The command object.
     * \return Returns true if the module was initialized successfully, otherwise false.
     */
    virtual bool processInit( WLEMMCommand::SPtr cmd );

    /**
     * Inherited method from WLEMMCommandProcessor.
     *
     * \param cmd The command object.
     * \return Returns true if the module was reseted successfully, otherwise false.
     */
    virtual bool processReset( WLEMMCommand::SPtr cmd );

private:
    enum CON_TYPE
    {
        CON_TCP, CON_UNIX
    };

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

    /**
     * Property group for the client
     */
    WPropGroup m_propGrpFtClient;

    /**
     * The selector for the connection type.
     */
    WPropSelection m_connectionTypeSelection;

    /**
     * A connection type selector item.
     */
    boost::shared_ptr< WItemSelection > m_connectionType;

    /**
     * The FieldTrip Buffer host.
     */
    WPropString m_host;

    /**
     * The FieldTrip Buffer host port.
     */
    WPropInt m_port;

    WPropInt m_blockSize;

    /**
     * The connection status.
     */
    WPropString m_conStatus;

    /**
     * The streaming status.
     */
    WPropString m_streamStatus;

    /**
     * Apply the data scaling.
     */
    WPropBool m_applyScaling;

    /**
     * The connect button.
     */
    WPropTrigger m_trgConnect;

    /**
     * The disconnect button.
     */
    WPropTrigger m_trgDisconnect;

    /**
     * The start streaming button.
     */
    WPropTrigger m_trgStartStream;

    /**
     * The stop streaming button.
     */
    WPropTrigger m_trgStopStream;

    /**
     * Trigger to reseting the module.
     */
    WPropTrigger m_resetModule;

    /**
     * The connection to the buffer.
     */
    WFTConnection::SPtr m_connection;

    /**
     * The FieldTrip streaming client
     */
    WFtbClient::SPtr m_ftRtClient;

    /**
     * Flag for stopping the streaming.
     */
    bool m_stopStreaming;

    /**
     * Method for updating the modules output connector and some GUI fields.
     *
     * \param emm The EMM object for delivering to the output connector.
     */
    void updateOutput( WLEMMeasurement::SPtr emm );

    /**
     * Callback when the connection type was changed.
     */
    void cbConnectionTypeChanged();

    /**
     * Handles a click on the connect button.
     *
     * \return Returns true if the callback was successfully, otherwise false.
     */
    bool hdlTrgConnect();

    /**
     * Handles a click on the disconnect button.
     */
    void hdlTrgDisconnect();

    /**
     * Callback, when the apply scaling checkboxes value was changed.
     */
    void cbApplyScaling();

    /**
     * Handles a click on the start streaming button.
     */
    void hdlTrgStartStreaming();

    /**
     * Callback when the stop streaming button was clicked.
     */
    void cbTrgStopStreaming();

    /**
     * Handles a click on the reset button.
     */
    void hdlTrgReset();

    /**
     * Switch the modules state after the client was connected to a FieldTrip Buffer server.
     */
    void applyStatusConnected();

    /**
     * Switch the modules state after the client was disconnected from a FieldTrip Buffer server.
     */
    void applyStatusDisconnected();

    /**
     * Switch the modules state after the client started the real-time streaming.
     */
    void applyStatusStreaming();

    /**
     * Switch the modules state after the client stopped the real-time streaming.
     */
    void applyStatusNotStreaming();
};

#endif  // WMFTRTCLIENT_H_
