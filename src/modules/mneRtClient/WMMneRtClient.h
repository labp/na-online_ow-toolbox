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

#ifndef WMMNERTCLIENT_H
#define WMMNERTCLIENT_H

#include <string>
#include <vector>

#include <core/common/WPropertyTypes.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLDigPoint.h"
#include "core/module/WLModuleDrawable.h"
#include "WRtClient.h"

/**
 * Streaming client for a MNE Real-time Server.
 *
 * \author pieloth
 * \ingroup io
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

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    // ---------------------------------
    // Methods for WLEMMCommandProcessor
    // ---------------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmd );
    virtual bool processReset( WLEMMCommand::SPtr cmd );

    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

private:
    WCondition::SPtr m_propCondition;

    /**
     * Property group for connection settings
     */
    WPropGroup m_propGrpConControl;

    WPropString m_propConIp;
    WPropString m_propConStatus;
    WPropTrigger m_trgConConnect;
    WPropTrigger m_trgConDisconnect;

    void hdlTrgConConnect();
    void hdlTrgConDisconnect();

    WPropString m_propDataStatus;
    WPropTrigger m_trgDataStart;
    WPropTrigger m_trgDataStop;

    void hdlTrgDataStart();
    void cbTrgDataStop();

    WRtClient::SPtr m_rtClient;
    bool m_stopStreaming;

    void hdlTrgConnectorChanged();
    WItemSelection::SPtr m_connectorItem;
    WPropSelection m_connectorSelection;

    WPropString m_simFile;
    WPropInt m_blockSize; //!< Samples of one block.

    void cbApplyScaling();
    WPropBool m_applyScaling;

    // Additional data //
    WPropGroup m_propGrpAdditional;

    WPropFilename m_digPointsFile;
    WLList< WLDigPoint >::SPtr m_digPoints;
    bool hdlDigPointsFileChanged( std::string fName );

    WPropString m_additionalStatus;

    /**
     * Reset additional infomation button.
     */
    WPropTrigger m_trgAdditionalReset;
    void hdlTrgAdditionalReset();
};

#endif  // WMMNERTCLIENT_H
