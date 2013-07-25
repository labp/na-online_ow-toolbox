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

#ifndef WMSOURCERECONSTRUCTION_H
#define WMSOURCERECONSTRUCTION_H

#include <string>

#include <core/common/WItemSelection.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"

#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleDrawable.h"
// TODO(pieloth): use OW classes
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WSourceReconstruction.h"

/**
 * This module implements several onscreen status displays
 * \ingroup modules
 */
class WMSourceReconstruction: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMSourceReconstruction();

    /**
     * destructor
     */
    virtual ~WMSourceReconstruction();

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

    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

private:
    // TODO(pieloth): use OW classes
    /**
     * Input connector for a EMM dataset
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropGroup m_propGrpSourceReconstruction;

    WPropBool m_useCuda;
    void handleImplementationChanged();

    // algorithm properties //
    WPropTrigger m_resetModule;
    WSourceReconstruction::SPtr m_sourceReconstruction;
    void handleResetTrigger();

    // Leadfield properties //
    WPropString m_leadfieldStatus;
    WPropInt m_leadfieldRows;
    WPropInt m_leadfieldCols;

    // Weighting properties //
    WItemSelection::SPtr m_weightingTypes;
    WPropSelection m_weightingTypesSelection;
    WPropString m_weightingStatus;
    WPropInt m_weightingRows;
    WPropInt m_weightingCols;
    void handleWeightingTypeChanged();

    // SNR properties //
    WPropDouble m_snr;
    void handleSnrChanged();

    LaBP::WEModalityType::Enum m_lastModality;
    void handleComputeModalityChanged( WLEMMCommand::ConstSPtr cmd );

    // Generate inverse solution //
    WPropString m_inverseStatus;
    WPropInt m_inverseRows;
    WPropInt m_inverseCols;
    bool inverseSolutionFromSubject( WLEMMeasurement::SPtr emm, LaBP::WEModalityType::Enum modality );

    double m_range;

    // data and noise covariance matices //
    WLMatrix::SPtr m_nCovarianceMatrix;
    WLMatrix::SPtr m_dCovarianceMatrix;

    // File status string //
    static const std::string NO_MATRIX_LOADED;
    static const std::string LOADING_MATRIX;
    static const std::string MATRIX_LOADED;
    static const std::string MATRIX_ERROR;
};

#endif  // WMSOURCERECONSTRUCTION_H
