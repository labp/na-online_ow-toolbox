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

#include "WBeamforming.h"

/**
 * This module implements several onscreen status displays.
 *
 * \author ehrlich
 * \ingroup analysis
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

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

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

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;
    WPropGroup m_propGrpBeamforming;
    WItemSelection::SPtr m_type;
    WPropSelection m_typeSelection;

    // Data covaiance
//    WPropFilename m_DataFile;
    WPropString m_DataStatus;
//    WLMatrix::SPtr m_Data;
//    bool handleDataChanged( std::string fName, WLMatrix::SPtr& data );

    WPropFilename m_CSDFile;
    WPropString m_CSDStatus;
    Eigen::MatrixXcd m_CSD;
    bool handleCSDChanged( std::string fName, Eigen::MatrixXcd* const csd );

//    WPropInt m_type;

    // algorithm properties
//    WPropTrigger m_resetModule;
    WBeamforming::SPtr m_beamforming;
//    void handleResetTrigger();

    // Modality
    WLEModality::Enum m_lastModality;
    void handleComputeModalityChanged();
    WPropBool m_useCuda;
    void handleImplementationChanged( void );
    WPropDouble m_reg;
};

#endif  // WMBEAMFORMING_H_
