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
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/WLTimeProfiler.h"

#include "WEpochSeparation.h"
#include "WMEpochSeparation.h"
#include "WMEpochSeparation.xpm"

using std::istringstream;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEpochSeparation )

WMEpochSeparation::WMEpochSeparation() :
                m_frequence( 1000 )
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
    return "Epoch Separation";
}

const std::string WMEpochSeparation::getDescription() const
{
    return "Extract samples around an event. Module supports LaBP data types only!";
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
    LaBP::WLModuleDrawable::properties();
    LaBP::WLModuleDrawable::setTimerangeInformationOnly( true );

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

void WMEpochSeparation::initModule()
{
    infoLog() << "Initializing module ...";
    waitRestored();
    initView( LaBP::WLEMDDrawable2D::WEGraphType::MULTI );
    m_separation = WEpochSeparation::SPtr( new WEpochSeparation() );
    handleResetTriggerPressed();
    infoLog() << "Initializing module finished!";
}

void WMEpochSeparation::moduleMain()
{
    LaBP::WLModuleDrawable::
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    WLEMMCommand::SPtr labpIn;
    LaBP::WLTimeProfiler::SPtr profiler( new LaBP::WLTimeProfiler( getName(), "process" ) );

    ready(); // signal ready state

    initModule();

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
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

        labpIn.reset();
        if( !m_input->isEmpty() )
        {
            labpIn = m_input->getData();
        }
        const bool dataValid = ( labpIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( labpIn );
        }
    }
}

void WMEpochSeparation::handleResetTriggerPressed()
{
    debugLog() << "handleResetTriggerPressed() called!";

    WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
    processReset( labp );

    m_resetTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    infoLog() << "Set new trigger values!";
}

bool WMEpochSeparation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    // TODO(pieloth): profiler

    WLEMMeasurement::SPtr emmOut;
//    LaBP::WLTimeProfiler::SPtr profilerIn;
    double frequence;
    // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
    debugLog() << "received data";

    if( emmIn->hasModality( this->getViewModality() ) )
    {
        frequence = emmIn->getModality( this->getViewModality() )->getSampFreq();
        if( frequence != m_frequence )
        {
            m_frequence = frequence;
            double samples = m_preTrigger->get() + m_postTrigger->get() + 1;
            this->setTimerange( samples / m_frequence );
        }
    }

//    profilerIn = emmIn->getTimeProfiler()->clone();
//    profilerIn->stop();
//    profiler->addChild( profilerIn );
//    if( !profiler->isStarted() )
//    {
//        profiler->start();
//    }

//    LaBP::WLTimeProfiler::SPtr trgProfiler = profiler->createAndAdd( WEpochSeparation::CLASS, "extract" );
//    trgProfiler->start();
    m_separation->extract( emmIn );
//    trgProfiler->stopAndLog();

    while( m_separation->hasEpochs() )
    {
        emmOut = m_separation->getNextEpoch();
        //emmOut->getTimeProfiler()->addChild( profiler );
        updateView( emmOut );
        //profiler->stopAndLog();

        WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        labp->setEmm( emmOut );
        m_output->updateData( labp );

        //profiler.reset( new LaBP::WLTimeProfiler( getName(), "process" ) );
    }
    return true;
}

bool WMEpochSeparation::processInit( WLEMMCommand::SPtr labp )
{
    // TODO(pieloth)
    m_output->updateData( labp );
    return false;
}

bool WMEpochSeparation::processReset( WLEMMCommand::SPtr labp )
{
    resetView();
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

    double samples = preSamples + postSamples + 1;
    this->setTimerange( samples / m_frequence );

    m_output->updateData( labp );

    return true;
}

