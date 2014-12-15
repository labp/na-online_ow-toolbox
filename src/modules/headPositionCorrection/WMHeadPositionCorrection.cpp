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

#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WMHeadPositionCorrection.h"
#include "WMHeadPositionCorrection.xpm"

W_LOADABLE_MODULE( WMHeadPositionCorrection )

WMHeadPositionCorrection::WMHeadPositionCorrection()
{
}

WMHeadPositionCorrection::~WMHeadPositionCorrection()
{
}

WModule::SPtr WMHeadPositionCorrection::factory() const
{
    WMHeadPositionCorrection::SPtr instance( new WMHeadPositionCorrection() );
    return instance;
}

const std::string WMHeadPositionCorrection::getName() const
{
    return WLConstantsModule::generateModuleName( "Head Position Correction" );
}

const std::string WMHeadPositionCorrection::getDescription() const
{
    return "Corrects the head position, MEG only (in progress).";
}

const char** WMHeadPositionCorrection::getXPMIcon() const
{
    return module_xpm;
}

void WMHeadPositionCorrection::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMHeadPositionCorrection::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = WCondition::SPtr( new WCondition() );
}

void WMHeadPositionCorrection::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // Wake up when input data changed
    m_moduleState.add( m_propCondition ); // Wake up when property changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    infoLog() << "Initializing module finished!";
}

void WMHeadPositionCorrection::moduleMain()
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
            break; // break mainLoop on shutdown
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );
        if( dataValid )
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}

bool WMHeadPositionCorrection::processCompute( WLEMMeasurement::SPtr emm )
{
    return false;
}

bool WMHeadPositionCorrection::processInit( WLEMMCommand::SPtr cmdIn )
{
    return false;
}

bool WMHeadPositionCorrection::processReset( WLEMMCommand::SPtr cmdIn )
{
    return false;
}
