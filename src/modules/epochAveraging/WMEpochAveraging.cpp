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
#include <typeinfo>
#include <vector>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WEpochAveraging.h"
#include "WEpochAveragingTotal.h"
#include "WEpochAveragingMoving.h"

#include "WMEpochAveraging.h"
#include "WMEpochAveraging.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEpochAveraging )

WMEpochAveraging::WMEpochAveraging()
{
}

WMEpochAveraging::~WMEpochAveraging()
{
}

WModule::SPtr WMEpochAveraging::factory() const
{
    return WModule::SPtr( new WMEpochAveraging() );
}

const char** WMEpochAveraging::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEpochAveraging::getName() const
{
    return WLConstantsModule::generateModuleName( "Epoch Averaging" );
}

const std::string WMEpochAveraging::getDescription() const
{
    return "Calculates moving or total average of incoming epochs.";
}

void WMEpochAveraging::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEpochAveraging::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = WCondition::SPtr( new WCondition() );

    // Averaging properties //
    const size_t tbase = 25;
    const size_t avgMovingSize = 10;

    m_propGrpAverage = m_properties->addPropertyGroup( "Averaging Properties", "Contains properties for averaging.", false );

    m_epochCount = m_propGrpAverage->addProperty( "Epoch count:", "Epoch count which are computed.", 0 );
    m_epochCount->setPurpose( PV_PURPOSE_INFORMATION );

    m_averageType = WItemSelection::SPtr( new WItemSelection() );

    WItemSelectionItemTyped< WEpochAveraging::SPtr >::SPtr item;
    WEpochAveraging::SPtr avg;

    avg.reset( new WEpochAveragingTotal( tbase ) );
    item.reset( new WItemSelectionItemTyped< WEpochAveraging::SPtr >( avg, "Total", "Computes total average." ) );
    m_averageType->addItem( item );

    avg.reset( new WEpochAveragingMoving( tbase, avgMovingSize ) );
    item.reset( new WItemSelectionItemTyped< WEpochAveraging::SPtr >( avg, "Moving", "Computes a moving average." ) );
    m_averageType->addItem( item );

    // getting the SelectorProperty from the list an add it to the properties
    m_averageTypeSelection = m_propGrpAverage->addProperty( "Average Type", "Choose a average type.",
                    m_averageType->getSelectorFirst(), boost::bind( &WMEpochAveraging::cbAverageTypeChanged, this ) );
    m_averageTypeSelection->changed( true );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_averageTypeSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_averageTypeSelection );

    m_tbase = m_propGrpAverage->addProperty( "TBase [samples]", "Sample count for baseline correction from index 0 to TBase.",
                    static_cast< int >( tbase ) );
    m_tbase->setMin( 0 );

    m_sizeMovingAverage = m_propGrpAverage->addProperty( "Moving Average [#epochs]",
                    "Number of epochs/trials for Moving Average.", static_cast< int >( avgMovingSize ) );

    m_resetAverage = m_propGrpAverage->addProperty( "(Re)set data", "(Re)set", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_resetAverage->changed( true );
}

void WMEpochAveraging::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    cbAverageTypeChanged();
    // handleResetAveragePressed(); ... called by callbackAverageTypeChanged

    infoLog() << "Restoring module finished!";
}

void WMEpochAveraging::moduleMain()
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
        if( m_resetAverage->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlResetAveragePressed();
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

void WMEpochAveraging::cbAverageTypeChanged()
{
    debugLog() << __func__ << "() called!";

    m_averaging = m_averageTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WEpochAveraging::SPtr > >()->getValue();
    if( typeid(WEpochAveragingTotal) == typeid(*m_averaging) )
    {
        m_sizeMovingAverage->setHidden( true );
    }
    else
    {
        m_sizeMovingAverage->setHidden( false );
    }
}

void WMEpochAveraging::hdlResetAveragePressed()
{
    debugLog() << __func__ << "() called!";

    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( cmd );

    m_resetAverage->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMEpochAveraging::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochAveraging", __func__ );
    WLEMMeasurement::SPtr emmOut;

    emmOut = m_averaging->getAverage( emmIn );

    const size_t count = m_averaging->getCount();
    debugLog() << "Averaging count: " << count;
    m_epochCount->set( count, true );

#ifdef DEBUG
    debugLog() << "Average pieces of first modality: ";
    size_t channels = emmOut->getModality( 0 )->getNrChans();
    size_t samples = emmOut->getModality( 0 )->getSamplesPerChan();
    for( size_t i = 0; i < 5 && i < channels; ++i )
    {
        std::stringstream ss;
        ss << "Channel " << i << ": ";
        for( size_t j = 0; j < 10 && j < samples; ++j )
        {
            ss << emmOut->getModality( 0 )->getData()( i, j ) << " ";
        }
        debugLog() << ss.str();
    }
#endif // DEBUG
    viewUpdate( emmOut );

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    labp->setEmm( emmOut );
    m_output->updateData( labp );

    return true;
}

bool WMEpochAveraging::processInit( WLEMMCommand::SPtr cmdIn )
{
    // TODO(pieloth): set block size?!
    m_output->updateData( cmdIn );
    return true;
}

bool WMEpochAveraging::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_input->clear();
    viewReset();
    m_averaging->reset();
    m_epochCount->set( m_averaging->getCount(), true );
    infoLog() << "Reset averaging!";
    m_averaging->setTBase( static_cast< size_t >( m_tbase->get() ), false );
    infoLog() << "Set tbase to " << m_averaging->getTBase();

    WEpochAveragingMoving::SPtr avgMov = m_averaging->getAs< WEpochAveragingMoving >();
    if( avgMov )
    {
        avgMov->setSize( m_sizeMovingAverage->get() );
        infoLog() << "Set moving average size to " << avgMov->getSize();
    }

    m_output->updateData( cmdIn );
    return true;
}
