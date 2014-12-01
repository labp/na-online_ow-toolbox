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

#ifndef WMEPOCHAVERAGING_H
#define WMEPOCHAVERAGING_H

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"

#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WEpochAveraging.h"

/**
 * Computes an average of all incoming EMDs.
 * Incoming EMDs must be stimuli-locked, i.e. so called epochs/trials.
 * This is not checked and reported by this module!
 *
 * \author pieloth
 * \ingroup preproc
 */
class WMEpochAveraging: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMEpochAveraging();

    /**
     * destructor
     */
    virtual ~WMEpochAveraging();

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
    virtual boost::shared_ptr< WModule > factory() const;

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
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    // Averaging properties //
    WPropGroup m_propGrpAverage;
    WPropTrigger m_resetAverage;

    WPropInt m_epochCount;

    /**
     * Resets the averaging algorithm.
     */
    void handleResetAveragePressed();
    WPropInt m_sizeMovingAverage;
    boost::shared_ptr< WItemSelection > m_averageType;
    WPropSelection m_averageTypeSelection;

    WPropInt m_tbase;

    /**
     * Switches the averaging algorithm.
     */
    void callbackAverageTypeChanged();

    WEpochAveraging::SPtr m_averaging;
};

#endif  // WMEPOCHAVERAGING_H
