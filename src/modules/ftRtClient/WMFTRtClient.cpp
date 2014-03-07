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

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

// Input & output data
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/util/profiler/WLTimeProfiler.h"

#include "WMFTRtClient.h"

#include "WMFTRtClient.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFTRtClient )

WMFTRtClient::WMFTRtClient()
{

}

WMFTRtClient::~WMFTRtClient()
{

}

boost::shared_ptr< WModule > WMFTRtClient::factory() const
{
    return boost::shared_ptr< WModule >( new WMFTRtClient() );
}

const char** WMFTRtClient::getXPMIcon() const
{
    return module_xpm;
}

/**
 * Returns the module name.
 */
const std::string WMFTRtClient::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " FieldTrip Real-time Client";
}

/**
 * Returns the module description.
 */
const std::string WMFTRtClient::getDescription() const
{
    return "Reads data for a FieldTrip Buffer and import them into Openwalnut. Module supports LaBP data types only!";
}

/**
 * Create the module connectors.
 */
void WMFTRtClient::connectors()
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

/**
 * Define the property panel.
 */
void WMFTRtClient::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::hideComputeModalitySelection( true );

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );
}

/**
 * Method for init the module.
 */
void WMFTRtClient::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::SINGLE );

    infoLog() << "Initializing module finished!";
}

void WMFTRtClient::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr emmIn;

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            m_moduleState.wait(); // wait for events like input-data or properties changed
        }

        // receive data form the input-connector
        emmIn.reset();
        if( !m_input->isEmpty() )
        {
            emmIn = m_input->getData();
        }
        const bool dataValid = ( emmIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the input-connector
        {
            // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
            debugLog() << "received data";

            process( emmIn );

            debugLog() << "finished";
        }
    }
}

bool WMFTRtClient::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMFTRtClient", "processCompute" );

    // show process visualization
    boost::shared_ptr< WProgress > rejectProcess = boost::shared_ptr< WProgress >(
                    new WProgress( "Import data from FieldTrip Buffer." ) );
    m_progress->addSubProgress( rejectProcess );

    // ---------- PROCESSING ----------
    viewUpdate( emmIn ); // update the GUI component



    rejectProcess->finish(); // finish the process visualization

    return true;
}

bool WMFTRtClient::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return false;
}

bool WMFTRtClient::processReset( WLEMMCommand::SPtr labp )
{
    viewReset();

    m_input->clear();
    m_output->updateData( labp );

    return true;
}
