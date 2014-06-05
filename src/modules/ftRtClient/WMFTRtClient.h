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

#ifndef WMFTRTCLIENT_H_
#define WMFTRTCLIENT_H_

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "fieldtrip/connection/WFTConnection.h"
#include "WFTNeuromagClient.h"

/**
 * The FieldTrip Real-time Client implements a streaming client from a FieldTrip Buffer server used by several EEG/ MEG acquisition systems.
 * The client receives data from the buffer server and computes them into the internal data strctures. After that the data will be send into
 * the processing chain of the OpenWalnutToolbox.
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
     * @param emm The measurement object.
     * @return Returns true if the computaion was successfully, otherwise false.
     */
    virtual bool processCompute( WLEMMeasurement::SPtr emm );

    /**
     * Inherited method from WLEMMCommandProcessor.
     *
     * @param labp The command object.
     * @return Returns true if the module was initialized successfully, otherwise false.
     */
    virtual bool processInit( WLEMMCommand::SPtr labp );

    /**
     * Inherited method from WLEMMCommandProcessor.
     *
     * @param labp The command object.
     * @return Returns true if the module was reseted successfully, otherwise false.
     */
    virtual bool processReset( WLEMMCommand::SPtr labp );

private:

    /**
     * Input connector for a EMM data set.
     */
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

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

    /**
     * The request timeout.
     */
    WPropInt m_waitTimeout;

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
     * Property group for header data
     */
    WPropGroup m_propGrpHeader;

    /**
     * The number of channels.
     */
    WPropInt m_channels;

    /**
     * The number of read samples.
     */
    WPropInt m_samples;

    /**
     * The used data type.
     */
    WPropString m_dataType;

    /**
     * The sampling frequency.
     */
    WPropDouble m_frSample;

    /**
     * The number of read events.
     */
    WPropInt m_events;

    /**
     * The size of the header structure.
     */
    WPropInt m_headerBufSize;

    /**
     * Property Group for additional information.
     */
    WPropGroup m_propGrpAdditionalInfo;

    /**
     * The load Source Space file button.
     */
    WPropFilename m_sourceSpaceFile;

    /**
     * The load BEM Layer button.
     */
    WPropFilename m_bemLayerFile;

    /**
     * The load Leadfield EEG button.
     */
    WPropFilename m_leadfieldEEGFile;

    /**
     * The load Leadfield MEG button.
     */
    WPropFilename m_leadfieldMEGFile;

    /**
     * The reset additional information button.
     */
    WPropTrigger m_trgAdditionalReset;

    /**
     * File status string.
     */
    WPropString m_additionalFileStatus;

    /**
     * The connection to the buffer.
     */
    WFTConnection::SPtr m_connection;

    /**
     * The FieldTrip streaming client
     */
    WFTNeuromagClient::SPtr m_ftRtClient;

    /**
     * The subject.
     */
    WLEMMSubject::SPtr m_subject;

    /**
     * The head surface information.
     */
    WLEMMSurface::SPtr m_surface;

    /**
     * The list of BEM boundaries.
     */
    WLList< WLEMMBemBoundary::SPtr >::SPtr m_bems;

    /**
     * The Leadfield EEG matrix.
     */
    WLMatrix::SPtr m_leadfieldEEG;

    /**
     * The Leadfield MEG matrix.
     */
    WLMatrix::SPtr m_leadfieldMEG;

    /**
     * Flag for stopping the streaming.
     */
    bool m_stopStreaming;

    /**
     * Method for updating the modules output connector and some GUI fields.
     *
     * @param emm The EMM object for delivering to the output connector.
     */
    void updateOutput( WLEMMeasurement::SPtr emm );

    /**
     * Callback when the connection type was changed.
     */
    void callbackConnectionTypeChanged();

    /**
     * Callback when the connect button was clicked.
     *
     * @return Returns true if the callback was successfully, otherwise false.
     */
    bool callbackTrgConnect();

    /**
     * Callback when the disconnect button was clicked.
     */
    void callbackTrgDisconnect();

    /**
     * Callback, when the apply scaling checkboxes value was changed.
     */
    void callbackApplyScaling();

    /**
     * Callback when the start streaming button was clicked.
     */
    void callbackTrgStartStreaming();

    /**
     * Callback when the stop streaming button was clicked.
     */
    void callbackTrgStopStreaming();

    /**
     * Callback when the reset button was clicked.
     */
    void callbackTrgReset();

    /**
     * Callback when a Source Space file was selected.
     *
     * @return Retruns true if the file was loaded, otherwise false.
     */
    bool callbackSourceSpace( std::string fName );

    /**
     * Callback when a BEM Layer file was selected.
     *
     * @return Retruns true if the file was loaded, otherwise false.
     */
    bool callbackBEMLayer( std::string fName );

    /**
     * Callback when a Leadfield EEG file was selected.
     *
     * @return Retruns true if the file was loaded, otherwise false.
     */
    bool callbackLeadfieldFile( std::string, WLMatrix::SPtr& leadfield );

    /**
     * Callback when the reset additional infomation button was clicked.
     */
    void callbackTrgAdditionalReset();

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

    /**
     * Shows the FieldTrip header structure in the GUI.
     */
    void dispHeaderInfo();

    /**
     * The default FieldTrip host name.
     */
    static const std::string DEFAULT_FT_HOST;

    /**
     * The default FieldTrip host port number.
     */
    static const int DEFAULT_FT_PORT;

    /**
     * The status string when connected.
     */
    static const std::string CONNECTION_CONNECT;

    /**
     * The status string when disconnected.
     */
    static const std::string CONNECTION_DISCONNECT;

    /**
     * The status string when streaming.
     */
    static const std::string CLIENT_STREAMING;

    /**
     * The status string when not streaming.
     */
    static const std::string CLIENT_NOT_STREAMING;

    /**
     * The status string when no file was loaded.
     */
    static const std::string NO_FILE_LOADED;

    /**
     * The status string during a file is loading.
     */
    static const std::string LOADING_FILE;

    /**
     * The status string when a file was loaded successfully.
     */
    static const std::string FILE_LOADED;

    /**
     * The status string when an error occured at the loading of a file.
     */
    static const std::string FILE_ERROR;

    /**
     * The standard path for file dialogs.
     */
    static const std::string STANDARD_FILE_PATH;
};

#endif /* WMFTRTCLIENT_H_ */
