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

#include <core/common/WPathHelper.h>

#include "core/data/WLEMMCommand.h"
#include "core/module/WLConstantsModule.h"

#include "WMHeadPositionEstimation.xpm"
#include "WMHeadPositionEstimation.h"

W_LOADABLE_MODULE( WMHeadPositionEstimation )

using LaBP::WLModuleInputDataRingBuffer;

WMHeadPositionEstimation::WMHeadPositionEstimation()
{
}

WMHeadPositionEstimation::~WMHeadPositionEstimation()
{
}

const std::string WMHeadPositionEstimation::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Head Position Estimation";
}

const std::string WMHeadPositionEstimation::getDescription() const
{
    return "TODO"; // TODO
}

WModule::SPtr WMHeadPositionEstimation::factory() const
{
    return WModule::SPtr( new WMHeadPositionEstimation );
}

const char** WMHeadPositionEstimation::getXPMIcon() const
{
    return module_xpm;
}

void WMHeadPositionEstimation::connectors()
{
    WModule::connectors();
    m_input.reset(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_input );
}

void WMHeadPositionEstimation::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    // TODO
    const std::string initial = "TODO";
    m_status = m_properties->addProperty( "Status:", "Status", initial );
    m_status->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMHeadPositionEstimation::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();
}

void WMHeadPositionEstimation::moduleMain()
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
        if( m_shutdownFlag() )
        {
            break;
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
            if( cmdIn->getCommand() == WLEMMCommand::Command::COMPUTE && cmdIn->hasEmm() )
            {
                ; // TODO
            }
        }
    }
}
