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

#ifndef WMBEAMFORMING_H_
#define WMBEAMFORMING_H_

#include <string>

#include <core/common/WItemSelection.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"

#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleDrawable.h"

#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"

#include "core/data/WLEMMBemBoundary.h"

/**
 * This module implements several onscreen status displays
 * \ingroup modules
 */
class WMBeamforming: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMBeamforming();

    /**
     * destructor
     */
    virtual ~WMBeamforming();

    /**
     * \par Description
     * Gives back the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * \par Description
     * Gives back a description of this module.
     * \return description to module.
     */
    virtual const std::string getDescription() const;

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual WModule::SPtr factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

protected:
    virtual void moduleInit();

    /**
     * \par Description
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

    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emmIn );
    virtual bool processInit( WLEMMCommand::SPtr cmdIn );
    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

private:

    // Input/Output connector for a EMM dataset

        WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;
        WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */
    /**
     * A condition used to notify about changes in several properties.
     */
        WCondition::SPtr m_propCondition;

        WPropGroup m_propGrpBeamforming;
        WPropFilename m_lfMEGFile;                                              //Für Button
        WPropString m_leadfieldStatus;
        WPropFilename m_BEMfile;
        WPropString m_BemStatus;
    // algorithm properties //
        WPropTrigger m_resetModule;
        WBeamforming::SPtr m_beamforming;
        void handleResetTrigger();

    //LEADFIELD
        bool handleLfFileChanged( std::string fName, WLMatrix::SPtr& lf );      //Arbeit mit fif File
        WLEMMSubject::SPtr m_subject;                                           //Leadfield
        WLMatrix::SPtr m_leadfieldMEG;                                          //LF Matrix

    //Modality
        WLEModality::Enum m_lastModality;
        void handleComputeModalityChanged( WLEMMCommand::ConstSPtr cmd );
        void handleImplementationChanged( void );
        WPropFilename m_bemFile;
        //WPropGroup m_propGrpAdditional;
       WLList< WLEMMBemBoundary::SPtr >::SPtr m_bems;
       bool handleBemFileChanged( std::string fName );
    WPropInt m_source;
};

#endif /* WMBEAMFORMING_H_ */
