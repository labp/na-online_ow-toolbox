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

#ifndef WMEPOCHSEPARATION_H
#define WMEPOCHSEPARATION_H

#include <string>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WEpochSeparation.h"

/**
 * Extracts an epoch/trial for an specified event/stimuli.
 *
 * \author pieloth
 * \ingroup preproc
 */
class WMEpochSeparation: public WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMEpochSeparation();

    /**
     * destructor
     */
    virtual ~WMEpochSeparation();

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

    // Trigger properties //
    WPropGroup m_propGrpTrigger;

    WPropInt m_preTrigger; //!< Samples before trigger/event.
    WPropInt m_postTrigger; //!< Samples after trigger/event.

    WPropString m_triggers;
    WPropInt m_triggerChannel;

    WPropTrigger m_resetTrigger;

    /**
     * Resets the epoch separation algorithm and sets the WProperties.
     */
    void hdlTrgResetPressed();

    WEpochSeparation::SPtr m_separation;
};

#endif  // WMEPOCHSEPARATION_H
