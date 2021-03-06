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

#include <string>
#include <sstream> // string to int
#include <set>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochSeparation.h"
#include "WMEpochSeparation.h"
#include "WMEpochSeparation.xpm"

using std::istringstream;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEpochSeparation )

WMEpochSeparation::WMEpochSeparation()
{
}

WMEpochSeparation::~WMEpochSeparation()
{
}

WModule::SPtr WMEpochSeparation::factory() const
{
    return WModule::SPtr( new WMEpochSeparation() );
}

const char** WMEpochSeparation::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEpochSeparation::getName() const
{
    return WLConstantsModule::generateModuleName( "Epoch Separation" );
}

const std::string WMEpochSeparation::getDescription() const
{
    return "Splits the continuous data stream according to event related responses into single epochs "
                    "that range from a time point to a point after stimulus onset.";
}

void WMEpochSeparation::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEpochSeparation::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = WCondition::SPtr( new WCondition() );

    // Trigger properties //
    m_propGrpTrigger = m_properties->addPropertyGroup( "Trigger Properties", "Contains properties for trigger.", false );

    m_preTrigger = m_propGrpTrigger->addProperty( "Pre [samples]", "Samples before trigger", 50 );
    m_postTrigger = m_propGrpTrigger->addProperty( "Post [samples]", "Samples after trigger", 100 );

    const std::string stdMask = "50";
    m_triggers = m_propGrpTrigger->addProperty( "Triggers", "Comma separated trigger values to detect.", stdMask );
    m_triggerChannel = m_propGrpTrigger->addProperty( "Event channel", "Event channel (counting from 0)", 0 );

    m_resetTrigger = m_propGrpTrigger->addProperty( "(Re)set", "(Re)set", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
}

void WMEpochSeparation::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::MULTI );
    m_separation = WEpochSeparation::SPtr( new WEpochSeparation() );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    hdlTrgResetPressed();

    infoLog() << "Restoring module finished!";
}

void WMEpochSeparation::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        // Configuration setup //
        if( m_resetTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgResetPressed();
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}

void WMEpochSeparation::hdlTrgResetPressed()
{
    debugLog() << __func__ << "() called!";

    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( cmd );

    m_resetTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    infoLog() << "Set new trigger values!";
}

bool WMEpochSeparation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochSeparation", __func__ );

    WLEMMeasurement::SPtr emmOut;

    m_separation->extract( emmIn );

    while( m_separation->hasEpochs() )
    {
        emmOut = m_separation->getNextEpoch();
        // Only update the view for the last epoch.

        WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
        labp->setEmm( emmOut );
        m_output->updateData( labp );
    }

    // Because updates are to fast to recognize changes, update the last EMM only.
    if( emmOut )
    {
        viewUpdate( emmOut );
    }
    return true;
}

bool WMEpochSeparation::processInit( WLEMMCommand::SPtr cmdIn )
{
    m_output->updateData( cmdIn );
    return true;
}

bool WMEpochSeparation::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_input->clear();
    viewReset();
    m_separation->reset();

    WLSampleNrT preSamples = m_preTrigger->get();
    WLSampleNrT postSamples = m_postTrigger->get();

    m_separation->setChannel( m_triggerChannel->get() );
    m_separation->setPreSamples( preSamples );
    m_separation->setPostSamples( postSamples );

    // Split string
    size_t current;
    size_t next = -1;
    const std::string strTriggers = m_triggers->get();
    std::string strTrigger;
    WLEMMeasurement::EventT trigger;
    std::set< WLEMMeasurement::EventT > triggers;
    do
    {
        current = next + 1;
        next = strTriggers.find_first_of( ",", current );
        strTrigger = strTriggers.substr( current, next - current );
        istringstream( strTrigger ) >> trigger;
        if( trigger != 0 )
        {
            triggers.insert( trigger );
            debugLog() << "Trigger: " << trigger;
        }
    } while( next != std::string::npos );
    infoLog() << "Trigger set: " << triggers.size();
    m_separation->setTriggerMask( triggers );

    m_output->updateData( cmdIn );

    return true;
}
