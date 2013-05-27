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

#ifndef WMEPOCHSEPARATION_H
#define WMEPOCHSEPARATION_H

#include <string>
#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WEpochSeparation.h"

/**
 * Module for epoch separation.
 * \ingroup modules
 */
class WMEpochSeparation: public LaBP::WLModuleDrawable
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

    // Trigger properties //
    WPropGroup m_propGrpTrigger;

    WPropInt m_preTrigger;
    WPropInt m_postTrigger;
    double m_frequence; // to calculate time rage

    WPropString m_triggers;
    WPropInt m_triggerChannel;

    WPropTrigger m_resetTrigger;

    /**
     * Resets the epoch separation algorithm and sets the WProperties.
     */
    void handleResetTriggerPressed();

    WEpochSeparation::SPtr m_separation;
};

#endif  // WMEPOCHSEPARATION_H
