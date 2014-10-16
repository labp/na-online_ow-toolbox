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
 * Module to parameterize and process a FIR filter.
 * \ingroup modules
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
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmdIn );
    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

private:
    // GUI event handler
    void handleDesignButtonPressed();
    void callbackFilterTypeChanged();
    void callbackCoeffFileChanged();
    void handleImplementationChanged();

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

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */

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
