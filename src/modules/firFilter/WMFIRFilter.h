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

#ifndef WMFIRFILTER_H
#define WMFIRFILTER_H

#include <string>

#include <core/kernel/WModule.h>

#include "core/dataHandler/WDataSetEMM.h"
#include "core/module/WLModuleDrawable.h"
// TODO(pieloth): use OW classes
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WFIRFilter.h"

/**
 * Module to parameterize and process a FIR filter.
 * \ingroup modules
 */
class WMFIRFilter: public LaBP::WLModuleDrawable
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

protected:
    virtual void initModule();

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
    virtual boost::shared_ptr< WModule > factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

private:
    // GUI event handler
    void callbackDesignButtonPressed();
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

    // TODO(pieloth): use OW classes
    /**
     * Input connector for a WEEG2 dataset to get filtered
     */
    boost::shared_ptr< LaBP::WLModuleInputDataRingBuffer< LaBP::WDataSetEMM > > m_input;

    /**
     * Output connector for a filtered WEEG2 dataset
     */
    boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM > > m_output;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

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
