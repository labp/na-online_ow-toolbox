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

#ifndef WMSOURCERECONSTRUCTION_H
#define WMSOURCERECONSTRUCTION_H

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

#include "WSourceReconstruction.h"

/**
 * Reconstruction of distributed sources using (weighted) minimum norm.
 *
 * \author pieloth
 * \ingroup analysis
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
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmdIn );
    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Buffered input connector.

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    /**
     * View property for bound calculator.
     */
    WPropInt m_percent;

    WPropGroup m_propGrpSourceReconstruction;

    WPropBool m_useCuda;
    void hdlImplementationChanged();

    // algorithm properties //
    WPropTrigger m_resetModule;
    WSourceReconstruction::SPtr m_sourceReconstruction;
    void hdlResetTrigger();

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
    void hdlWeightingTypeChanged();

    // SNR properties //
    WPropDouble m_snr;
    void hdlSnrChanged();

    WLEModality::Enum m_lastModality;
    void hdlComputeModalityChanged();

    // Generate inverse solution //
    WPropString m_inverseStatus;
    WPropInt m_inverseRows;
    WPropInt m_inverseCols;
    bool inverseSolutionFromSubject( WLEMMeasurement::SPtr emm, WLEModality::Enum modality );

    // data and noise covariance matices //
    WLMatrix::SPtr m_nCovarianceMatrix;
    WLMatrix::SPtr m_dCovarianceMatrix;

    void cbIncludesChanged();
};

#endif  // WMSOURCERECONSTRUCTION_H
