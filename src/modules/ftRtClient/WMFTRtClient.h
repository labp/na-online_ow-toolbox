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

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"

#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WFTClientStreaming.h"

class WMFTRtClient: public WLModuleDrawable
{
public:

    WMFTRtClient();
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

    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

private:

    /**
     * Input connector for a EMM data set.
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

    /**
     * Property group for the client
     */
    WPropGroup m_propGrpFtClient;

    WPropTrigger m_doRequest;

    boost::shared_ptr< WItemSelection > m_connectionType;
    WPropSelection m_connectionTypeSelection;

    WPropString m_host;
    WPropInt m_port;
    WPropString m_conStatus;
    WPropString m_streamStatus;

    WPropTrigger m_trgConnect;
    WPropTrigger m_trgDisconnect;
    WPropTrigger m_trgStartStream;
    WPropTrigger m_trgStopStream;

    /**
     * Property group for header data
     */
    WPropGroup m_propGrpHeader;

    WPropInt m_channels;
    WPropInt m_samples;
    WPropDouble m_frSample;
    WPropInt m_events;
    WPropInt m_headerBufSize;

    WFTClientStreaming::SPtr m_ftRtClient;

    void handleDoRequest();

    void callbackConnectionTypeChanged();

    void callbackTrgConnect();

    void callbackTrgDisconnect();

    void callbackTrgStartStreaming();

    void callbackTrgStopStreaming();

    static const std::string DEFAULT_FT_HOST;

    static const int DEFAULT_FT_PORT;

    static const std::string CONNECTION_CONNECT;

    static const std::string CONNECTION_DISCONNECT;

    static const std::string CLIENT_STREAMING;

    static const std::string CLIENT_NOT_STREAMING;
};

#endif /* WMFTRTCLIENT_H_ */
