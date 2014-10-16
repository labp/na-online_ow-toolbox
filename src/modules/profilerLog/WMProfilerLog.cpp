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

#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLLifetimeProfiler.h"
#include "core/util/profiler/WLProfilerLogger.h"

#include "WMProfilerLog.h"
#include "WMProfilerLog.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMProfilerLog )

WMProfilerLog::WMProfilerLog()
{
}

WMProfilerLog::~WMProfilerLog()
{
}

WModule::SPtr WMProfilerLog::factory() const
{
    return WModule::SPtr( new WMProfilerLog() );
}

const char** WMProfilerLog::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMProfilerLog::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Profile Logger";
}

const std::string WMProfilerLog::getDescription() const
{
    return "Just prints the LifetimeProfiler. (LaBP data types only)";
}

void WMProfilerLog::connectors()
{
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMProfilerLog::properties()
{
    WModule::properties();
}

void WMProfilerLog::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed

    WLEMMCommand::SPtr labpIn;

    ready(); // signal ready state

    debugLog() << "Entering main loop";

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

        labpIn.reset();
        if( !m_input->isEmpty() )
        {
            labpIn = m_input->getData();
        }
        const bool dataValid = ( labpIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid && labpIn->hasEmm() ) // If there was an update on the inputconnector
        {
            wlprofiler::log() << *( labpIn->getEmm()->getProfiler() );

            m_output->updateData( labpIn );
        }
    }
}
