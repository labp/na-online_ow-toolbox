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

#ifndef WMMNERTCLIENT_H
#define WMMNERTCLIENT_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WPropertyTypes.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMSurface.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/module/WLModuleDrawable.h"
#include "core/io/WLReaderExperiment.h"
#include "WRtClient.h"

/**
 * TODO
 * \ingroup modules
 */
class WMMneRtClient: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMMneRtClient();

    /**
     * destructor
     */
    virtual ~WMMneRtClient();

    /**
     * Returns the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * Returns a description of this module.
     * \return description of module.
     */
    virtual const std::string getDescription() const;

protected:
    // ---------------------------------
    // Methods for WLEMMCommandProcessor
    // ---------------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

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
    virtual WModule::SPtr factory() const;

    /**
     * Get the icon for this module in XPM format.
     * \return The icon.
     */
    virtual const char** getXPMIcon() const;

private:
    //! a condition for the matrix selection
    WCondition::SPtr m_propCondition;

    /**
     * Property group for connection settings
     */
    WPropGroup m_propGrpConControl;

    WPropString m_propConIp;
    WPropString m_propConStatus;
    WPropTrigger m_trgConConnect;
    WPropTrigger m_trgConDisconnect;

    void handleTrgConConnect();
    void handleTrgConDisconnect();

    static const std::string STATUS_CON_CONNECTED;
    static const std::string STATUS_CON_DISCONNECTED;
    static const std::string STATUS_CON_ERROR;

    WPropString m_propDataStatus;
    WPropTrigger m_trgDataStart;
    WPropTrigger m_trgDataStop;

    void handleTrgDataStart();
    void callbackTrgDataStop();

    WRtClient::SPtr m_rtClient;
    bool m_stopStreaming;

    void handleTrgConnectorChanged();
    WItemSelection::SPtr m_connectorItem;
    WPropSelection m_connectorSelection;
    static const int NO_CONNECTOR;

    WPropString m_simFile;

    static const std::string STATUS_DATA_STREAMING;
    static const std::string STATUS_DATA_ERROR;
    static const std::string STATUS_DATA_NOT_STREAMING;

    // Experiment loader //
    WPropGroup m_propGrpExperiment;

    LaBP::WLEMMSubject::SPtr m_subject;

    bool m_isExpLoaded;
    bool m_isFiffLoaded;

    WPropFilename m_fiffFile;
    WPropString m_fiffFileStatus;

    WPropString m_expSubject;

    WItemSelection::SPtr m_expBemFiles;
    WPropSelection m_expBemFilesSelection;

    WItemSelection::SPtr m_expSurfaces;
    WPropSelection m_expSurfacesSelection;

    WPropString m_expTrial;

    WPropTrigger m_expLoadTrigger;
    WPropString m_expLoadStatus;
    void handleExperimentLoadChanged();
    void handleExtractExpLoader( std::string fiffFile );
    WLReaderExperiment::SPtr m_expReader;

    // Additional data //
    WPropGroup m_propGrpAdditional;

    WPropFilename m_srcSpaceFile;
    LaBP::WLEMMSurface::SPtr m_surface;
    bool handleSurfaceFileChanged( std::string fName );

    WPropFilename m_bemFile;
    boost::shared_ptr< std::vector< LaBP::WLEMMBemBoundary::SPtr > > m_bems;
    bool handleBemFileChanged( std::string fName );

    WPropFilename m_lfEEGFile;
    WPropFilename m_lfMEGFile;
    WLMatrix::SPtr m_leadfieldEEG;
    WLMatrix::SPtr m_leadfieldMEG;
    bool handleLfFileChanged( std::string fName, WLMatrix::SPtr& lf );

    WPropString m_additionalStatus;

    // File status string //
    static const std::string NO_DATA_LOADED;
    static const std::string LOADING_DATA;
    static const std::string DATA_LOADED;
    static const std::string DATA_ERROR;
    static const std::string NO_FILE_LOADED;
    static const std::string LOADING_FILE;
    static const std::string FILE_LOADED;
    static const std::string FILE_ERROR;
};

const std::string WMMneRtClient::STATUS_CON_CONNECTED = "Connected";
const std::string WMMneRtClient::STATUS_CON_DISCONNECTED = "Disconnected";
const std::string WMMneRtClient::STATUS_CON_ERROR = "Error";

const std::string WMMneRtClient::STATUS_DATA_STREAMING = "Streaming";
const std::string WMMneRtClient::STATUS_DATA_ERROR = "Error";
const std::string WMMneRtClient::STATUS_DATA_NOT_STREAMING = "Not streaming";

const std::string WMMneRtClient::NO_FILE_LOADED = "No file loaded.";
const std::string WMMneRtClient::LOADING_FILE = "Loading file ...";
const std::string WMMneRtClient::FILE_LOADED = "File successfully loaded.";
const std::string WMMneRtClient::FILE_ERROR = "Could not load file.";

const std::string WMMneRtClient::NO_DATA_LOADED = "No data loaded.";
const std::string WMMneRtClient::LOADING_DATA = "Loading data ...";
const std::string WMMneRtClient::DATA_LOADED = "Data successfully loaded.";
const std::string WMMneRtClient::DATA_ERROR = "Could not load data.";

#endif  // WMMNERTCLIENT_H
