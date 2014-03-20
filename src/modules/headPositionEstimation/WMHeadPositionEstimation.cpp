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
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WMHeadPositionEstimation.xpm"
#include "WMHeadPositionEstimation.h"

W_LOADABLE_MODULE( WMHeadPositionEstimation )

using LaBP::WLModuleInputDataRingBuffer;
using LaBP::WLModuleOutputDataCollectionable;

const double WMHeadPositionEstimation::HPI1_FREQ = 154.0;
const double WMHeadPositionEstimation::HPI2_FREQ = 158.0;
const double WMHeadPositionEstimation::HPI3_FREQ = 162.0;
const double WMHeadPositionEstimation::HPI4_FREQ = 166.0;
const double WMHeadPositionEstimation::HPI5_FREQ = 170.0;

const double WMHeadPositionEstimation::WINDOWS_SIZE = 200.0;
const double WMHeadPositionEstimation::STEP_SIZE = 10.0;

const std::string WMHeadPositionEstimation::STATUS_OK = "Ok";
const std::string WMHeadPositionEstimation::STATUS_ERROR = "Error";

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
    return "Estimation of the head position using the continuous signals of HPI coils.";
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
    m_input.reset( new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in", "Incoming WLEMMComand" ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out", "Outgoing WLEMMComand" ) );
    addConnector( m_output );
}

void WMHeadPositionEstimation::properties()
{
    WLModuleDrawable::properties();

    m_condition = WCondition::SPtr( new WCondition() );

    m_propGroup = m_properties->addPropertyGroup( "Head Position Estimation", "HeadPositionEstimation" );

    m_propHpi1Freq = m_propGroup->addProperty( "HPI #1 Frequency [Hz]:", "Frequency of HPI coil 1 in Hertz.", HPI1_FREQ );
    m_propHpi2Freq = m_propGroup->addProperty( "HPI #2 Frequency [Hz]:", "Frequency of HPI coil 2 in Hertz.", HPI2_FREQ );
    m_propHpi3Freq = m_propGroup->addProperty( "HPI #3 Frequency [Hz]:", "Frequency of HPI coil 3 in Hertz.", HPI3_FREQ );
    m_propHpi4Freq = m_propGroup->addProperty( "HPI #4 Frequency [Hz]:", "Frequency of HPI coil 4 in Hertz.", HPI4_FREQ );
    m_propHpi5Freq = m_propGroup->addProperty( "HPI #5 Frequency [Hz]:", "Frequency of HPI coil 5 in Hertz.", HPI5_FREQ );

    m_propWindowsSize = m_propGroup->addProperty( "Windows Size [ms]:", "Windows size in milliseconds.", WINDOWS_SIZE );
    m_propStepSize = m_propGroup->addProperty( "Step Size [ms]:", "Step size in milliseconds.", STEP_SIZE );

    m_propStatus = m_propGroup->addProperty( "Status:", "Reports the status of actions.", STATUS_OK );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgApplySettings = m_propGroup->addProperty( "Apply Settings:", "Apply", WPVBaseTypes::PV_TRIGGER_READY, m_condition );
}

void WMHeadPositionEstimation::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_condition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::MULTI );

    infoLog() << "Initializing module finished!";
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

        if( m_trgApplySettings->changed() )
        {
            handleApplyFreq();
            m_trgApplySettings->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
            process( cmdIn );
        }
    }

    viewCleanup();
}

bool WMHeadPositionEstimation::handleApplyFreq()
{
    m_propStatus->set( STATUS_ERROR, true );
    return false;
}

bool WMHeadPositionEstimation::processInit( WLEMMCommand::SPtr cmdIn )
{
    const bool rc = handleApplyFreq();
    m_output->updateData( cmdIn );
    return rc;
}

bool WMHeadPositionEstimation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    // TODO (pieloth): implement
    WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmdOut->setEmm( emmIn );
    m_output->updateData( cmdOut );
    return false;
}

bool WMHeadPositionEstimation::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_output->updateData( cmdIn );
    return true;
}
