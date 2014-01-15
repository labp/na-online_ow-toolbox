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

#include <string>
#include <sstream> // string to int
#include <set>

#include <boost/shared_ptr.hpp>

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

boost::shared_ptr< WModule > WMEpochSeparation::factory() const
{
    return boost::shared_ptr< WModule >( new WMEpochSeparation() );
}

const char** WMEpochSeparation::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEpochSeparation::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Epoch Separation";
}

const std::string WMEpochSeparation::getDescription() const
{
    return "Splits the continuous data stream according to event related responses into single epochs "
                    "that range from a time point to a point after stimulus onset.";
}

void WMEpochSeparation::connectors()
{
    m_input = LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMEpochSeparation::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    // Trigger properties //
    m_propGrpTrigger = m_properties->addPropertyGroup( "Trigger Properties", "Contains properties for trigger.", false );

    m_preTrigger = m_propGrpTrigger->addProperty( "Pre-Samples", "Samples before trigger", 50 );
    m_postTrigger = m_propGrpTrigger->addProperty( "Post-Samples", "Samples after trigger", 100 );

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

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::MULTI );
    m_separation = WEpochSeparation::SPtr( new WEpochSeparation() );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    handleResetTriggerPressed();

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
            handleResetTriggerPressed();
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
}

void WMEpochSeparation::handleResetTriggerPressed()
{
    debugLog() << "handleResetTriggerPressed() called!";

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( labp );

    m_resetTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    infoLog() << "Set new trigger values!";
}

bool WMEpochSeparation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochSeparation", "processCompute" );

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
    viewReset();
    m_separation->reset();

    int preSamples = m_preTrigger->get();
    int postSamples = m_postTrigger->get();

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

