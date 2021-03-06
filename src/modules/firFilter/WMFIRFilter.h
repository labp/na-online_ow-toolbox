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

#ifndef WMFIRFILTER_H
#define WMFIRFILTER_H

#include <string>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WFIRFilter.h"

/**
 * FIR Filter for lowpass, highpass, bandpass and bandstop filtering of EEG/MEG data.
 * \see \cite Kaehler2011
 *
 * \authors kaehler, pieloth
 * \ingroup preproc
 */
class WMFIRFilter: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMFIRFilter();

    /**
     * destructor
     */
    virtual ~WMFIRFilter();

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
    // GUI event handler
    void hdlDesignButtonPressed();
    void cbFilterTypeChanged();
    void cbCoeffFileChanged();
    void hdlImplementationChanged();

    WPropGroup m_propGrpFirFilter;

    WFIRFilter::SPtr m_firFilter;

    WPropDouble m_cFreq1;
    WPropDouble m_cFreq2;
    WPropInt m_order;
    WItemSelection::SPtr m_windows;
    WItemSelection::SPtr m_filterTypes;
    WPropSelection m_filterTypeSelection;
    WPropSelection m_windowSelection;
    WPropDouble m_samplingFreq;
    WPropTrigger m_designTrigger;

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Buffered input connector.

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    /**
     * Exported MatLAB File with coefficients
     */
    WPropFilename m_coeffFile;

    /**
     * The property to know whether use CUDA or not while runtime
     */
    WPropBool m_useCuda;
};

#endif  // WMFIRFILTER_H
