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

    // Averaging properties //
    WPropGroup m_propGrpAverage;
    WPropTrigger m_resetAverage;

    WPropInt m_epochCount;

    /**
     * Resets the averaging algorithm.
     */
    void hdlResetAveragePressed();
    WPropInt m_sizeMovingAverage; //!< Size of moving average in epochs/trials.
    WItemSelection::SPtr m_averageType;
    WPropSelection m_averageTypeSelection;

    WPropInt m_tbase; //!< Samples for baseline correction.

    /**
     * Switches the averaging algorithm.
     */
    void cbAverageTypeChanged();

    WEpochAveraging::SPtr m_averaging;
};

#endif  // WMEPOCHAVERAGING_H
