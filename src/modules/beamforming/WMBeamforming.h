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

//#include "core/data/WLEMMBemBoundary.h"

/**
 * This module implements several onscreen status displays.
 *
 * \ingroup modules
 * \author ehrlich
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
    //kein outputdefinieren, ist in draw definiert, nur den connector setzen
    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;
    WPropGroup m_propGrpBeamforming;
    WItemSelection::SPtr m_type;
    WPropFilename m_lfEEGFile;
    WPropString m_leadfieldStatus;
//Leadfield
    bool handleLfFileChanged( std::string fName, WLMatrix::SPtr& lf );      //Arbeit mit fif/mat File
    WLEMMSubject::SPtr m_subject;                                           //Leadfield
    //WLMatrix::SPtr m_leadfieldMEG;                                          //LF Matrix
    WLMatrix::SPtr m_leadfieldEEG;
    WPropSelection m_typeSelection;

//Data covaiance
//     WPropFilename m_DataFile;
     WPropString m_DataStatus;
//     WLMatrix::SPtr m_Data;
//     bool handleDataChanged( std::string fName, WLMatrix::SPtr& data );


      WPropFilename m_CSDFile;
      WPropString m_CSDStatus;
      Eigen::MatrixXcd m_CSD;
      bool handleCSDChanged( std::string fName, Eigen::MatrixXcd* const csd );

////Type
//      WPropInt m_type;

// algorithm properties
//    WPropTrigger m_resetModule;
    WBeamforming::SPtr m_beamforming;
//    void handleResetTrigger();

//Modality
    WLEModality::Enum m_lastModality;
    void handleComputeModalityChanged( WLEMMCommand::ConstSPtr cmd );
    WPropBool m_useCuda;
    void handleImplementationChanged( void );
    WPropDouble m_reg;



};

#endif /* WMBEAMFORMING_H_ */
