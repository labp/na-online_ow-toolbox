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
#include <typeinfo>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
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

boost::shared_ptr< WModule > WMEpochAveraging::factory() const
{
    return boost::shared_ptr< WModule >( new WMEpochAveraging() );
}

const char** WMEpochAveraging::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEpochAveraging::getName() const
{
    return "Epoch Averaging";
}

const std::string WMEpochAveraging::getDescription() const
{
    return "Calculates the average of the input epochs. Module supports LaBP data types only!";
}

void WMEpochAveraging::connectors()
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

void WMEpochAveraging::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    // Averaging properties //
    const size_t tbase = 25;
    const size_t avgMovingSize = 10;

    m_propGrpAverage = m_properties->addPropertyGroup( "Averaging Properties", "Contains properties for averaging.", false );

    m_epochCount = m_propGrpAverage->addProperty( "Epoch count:", "Epoch count which are computed.", 0 );
    m_epochCount->setPurpose( PV_PURPOSE_INFORMATION );

    m_averageType = boost::shared_ptr< WItemSelection >( new WItemSelection() );

    boost::shared_ptr< WItemSelectionItemTyped< WEpochAveraging::SPtr > > item;
    WEpochAveraging::SPtr avg;

    avg.reset( new WEpochAveragingTotal( tbase ) );
    item.reset( new WItemSelectionItemTyped< WEpochAveraging::SPtr >( avg, "Total", "Computes total average." ) );
    m_averageType->addItem( item );

    avg.reset( new WEpochAveragingMoving( tbase, avgMovingSize ) );
    item.reset( new WItemSelectionItemTyped< WEpochAveraging::SPtr >( avg, "Moving", "Computes a moving average." ) );
    m_averageType->addItem( item );

    // getting the SelectorProperty from the list an add it to the properties
    m_averageTypeSelection = m_propGrpAverage->addProperty( "Average Type", "Choose a average type.",
                    m_averageType->getSelectorFirst(), boost::bind( &WMEpochAveraging::callbackAverageTypeChanged, this ) );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_averageTypeSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_averageTypeSelection );

    m_tbase = m_propGrpAverage->addProperty( "TBase", "Sample count for baseline correction from index 0 to TBase.",
                    static_cast< int >( tbase ) );
    m_tbase->setMin( 0 );

    m_sizeMovingAverage = m_propGrpAverage->addProperty( "Size of Moving Average", "Size of Moving Average in samples",
                    static_cast< int >( avgMovingSize ) );

    m_resetAverage = m_propGrpAverage->addProperty( "(Re)set data", "(Re)set", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
}

void WMEpochAveraging::moduleInit()
{
    infoLog() << "Initializing module ...";
    waitRestored();

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::SINGLE );

    callbackAverageTypeChanged();

    infoLog() << "Initializing module finished!";
}

void WMEpochAveraging::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    WLEMMCommand::SPtr labpIn;

    ready(); // signal ready state

    moduleInit();

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

        // Configuration setup //
        if( m_resetAverage->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleResetAveragePressed();
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

void WMEpochAveraging::callbackAverageTypeChanged()
{
    debugLog() << "handleAverageTypeChanged() called!";

    m_averaging = m_averageTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WEpochAveraging::SPtr > >()->getValue();
    if( typeid(WEpochAveragingTotal) == typeid(*m_averaging) )
    {
        m_sizeMovingAverage->setHidden( true );
    }
    else
    {
        m_sizeMovingAverage->setHidden( false );
    }
    handleResetAveragePressed();
}

void WMEpochAveraging::handleResetAveragePressed()
{
    debugLog() << "handleResetAveragePressed() called!";

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
    processReset( labp );

    m_resetAverage->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMEpochAveraging::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMEpochAveraging", "processCompute" );
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

bool WMEpochAveraging::processInit( WLEMMCommand::SPtr labp )
{
    // TODO(pieloth)
    m_output->updateData( labp );
    return false;
}

bool WMEpochAveraging::processReset( WLEMMCommand::SPtr labp )
{
    viewReset();
    m_averaging->reset();
    m_epochCount->set( m_averaging->getCount(), true );
    infoLog() << "Reset averaging!";
    m_averaging->setTBase( static_cast< size_t >( m_tbase->get() ), false );
    infoLog() << "Set tbase to " << m_averaging->getTBase();

    WEpochAveragingMoving::SPtr avgMov = boost::shared_dynamic_cast< WEpochAveragingMoving >( m_averaging );
    if( avgMov )
    {
        avgMov->setSize( m_sizeMovingAverage->get() );
        infoLog() << "Set moving average size to " << avgMov->getSize();
    }

    m_input->clear();
    m_output->updateData( labp );
    return false;
}
