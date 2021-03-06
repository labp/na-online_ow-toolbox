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

#include <algorithm>    // std::max
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <core/common/WException.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/preprocessing/WLWindowFunction.h"
#include "core/util/profiler/WLTimeProfiler.h"

// FIR filter implementations
#include "WFIRFilter.h"
#ifdef FOUND_CUDA
#include "WFIRFilterCuda.h"
#endif //FOUND_CUDA
#include "WFIRFilterCpu.h"

#include "WMFIRFilter.h"
#include "WMFIRFilter.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMFIRFilter )

WMFIRFilter::WMFIRFilter()
{
}

WMFIRFilter::~WMFIRFilter()
{
}

WModule::SPtr WMFIRFilter::factory() const
{
    return WModule::SPtr( new WMFIRFilter() );
}

const char** WMFIRFilter::getXPMIcon() const
{
    return firfilter_xpm;
}

const std::string WMFIRFilter::getName() const
{
    return WLConstantsModule::generateModuleName( "FIR Filter" );
}

const std::string WMFIRFilter::getDescription() const
{
    return "Filters the signals according to lowpass, highpass, bandpass and bandstop characteristic.";
}

void WMFIRFilter::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMFIRFilter::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::hideComputeModalitySelection( true );

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propGrpFirFilter = m_properties->addPropertyGroup( "FIR Filter", "Contains properties for FIR Filter.", false );

    m_coeffFile = m_propGrpFirFilter->addProperty( "Coefficients:", "Load coefficients from file.", WPathHelper::getAppPath(),
                    boost::bind( &WMFIRFilter::cbCoeffFileChanged, this ) );
    m_coeffFile->changed( true );

    m_useCuda = m_propGrpFirFilter->addProperty( "Use Cuda", "Activate CUDA support.", true, m_propCondition );
#ifndef FOUND_CUDA
    m_useCuda->setHidden( true );
#endif // FOUND_CUDA
    m_useCuda->changed( true );

    // creating the list of Filtertypes
    m_filterTypes = WItemSelection::SPtr( new WItemSelection() );
    const std::set< WFIRFilter::WEFilterType::Enum > fEnums = WFIRFilter::WEFilterType::values();
    for( std::set< WFIRFilter::WEFilterType::Enum >::const_iterator it = fEnums.begin(); it != fEnums.end(); ++it )
    {
        m_filterTypes->addItem(
                        WItemSelectionItemTyped< WFIRFilter::WEFilterType::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WFIRFilter::WEFilterType::Enum >( *it,
                                                        WFIRFilter::WEFilterType::name( *it ),
                                                        WFIRFilter::WEFilterType::name( *it ) ) ) );
    }

    // getting the SelectorProperty from the list an add it to the properties
    m_filterTypeSelection = m_propGrpFirFilter->addProperty( "FilterType", "What kind of filter do you want to use",
                    m_filterTypes->getSelectorFirst(), boost::bind( &WMFIRFilter::cbFilterTypeChanged, this ) );

    // Be sure it is at least one selected, but not more than one
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_filterTypeSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_filterTypeSelection );

    // same with windows
    m_windows = WItemSelection::SPtr( new WItemSelection() );
    const std::set< WLWindowFunction::WLEWindow > wEnums = WLWindowFunction::values();
    for( std::set< WLWindowFunction::WLEWindow >::const_iterator it = wEnums.begin(); it != wEnums.end(); ++it )
    {
        m_windows->addItem(
                        WItemSelectionItemTyped< WLWindowFunction::WLEWindow >::SPtr(
                                        new WItemSelectionItemTyped< WLWindowFunction::WLEWindow >( *it,
                                                        WLWindowFunction::name( *it ), WLWindowFunction::name( *it ) ) ) );
    }

    m_windowSelection = m_propGrpFirFilter->addProperty( "Window",
                    "What kind of window do you want to use for designing a filter", m_windows->getSelectorFirst(),
                    m_propCondition );

    WPropertyHelper::PC_SELECTONLYONE::addTo( m_windowSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_windowSelection );

    // the frequencies
    m_samplingFreq = m_propGrpFirFilter->addProperty( "Sampling Frequency [Hz]",
                    "Samplingfrequency comes from data. Do only change this for down- or upsampling", 500.0 );
    m_samplingFreq->setMin( 1.0 );
    m_samplingFreq->setMax( 16000.0 );

    m_cFreq1 = m_propGrpFirFilter->addProperty( "Cutoff frequency 1 [Hz]", "Frequency for filterdesign", 1.0 );
    m_cFreq1->setMin( 0.0 );
    m_cFreq1->setMax( 2000.0 );

    m_cFreq2 = m_propGrpFirFilter->addProperty( "Cutoff frequency 2 [Hz]",
                    "Frequency for filterdesign. Second frequency is needed for bandpass and bandstop", 20.0 );
    m_cFreq2->setMin( 0.0 );
    m_cFreq2->setMax( 2000.0 );
    m_cFreq2->setHidden( true );

    m_order = m_propGrpFirFilter->addProperty( "Order:", "The number of coeffitients depends on the order", 200 );
    m_order->setMax( 5000 );

    // button for starting design
    m_designTrigger = m_propGrpFirFilter->addProperty( "Filter:", "Calculate Filtercoeffitients", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_designTrigger->changed( true );
}

void WMFIRFilter::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // Wake up when input data changed
    m_moduleState.add( m_propCondition ); // Wake up when property changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    hdlImplementationChanged();
    cbFilterTypeChanged();
    if( !( m_coeffFile->get().string().empty() )
                    && m_coeffFile->get().string().compare( WPathHelper::getAppPath().string() ) != 0 )
    {
        cbCoeffFileChanged();
    }
    hdlDesignButtonPressed();

    infoLog() << "Restoring module finished!";
}

void WMFIRFilter::moduleMain()
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

        if( m_useCuda->changed( true ) )
        {
            hdlImplementationChanged();
        }

        if( m_designTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlDesignButtonPressed();
            WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::RESET );
            processReset( cmd );
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

void WMFIRFilter::cbCoeffFileChanged( void )
{
    debugLog() << __func__ << "() called!";
    const char *path = m_coeffFile->get().string().c_str();
    infoLog() << "Reading *.fcf file: " << m_coeffFile->get().string();

    m_firFilter->setCoefficients( path );
}

void WMFIRFilter::hdlImplementationChanged( void )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Changing FIR Filter" ) );
    m_progress->addSubProgress( progress );

    WFIRFilter::WEFilterType::Enum fType = m_filterTypeSelection->get().at( 0 )->getAs<
                    WItemSelectionItemTyped< WFIRFilter::WEFilterType::Enum > >()->getValue();
    WLWindowFunction::WLEWindow wType = m_windowSelection->get().at( 0 )->getAs<
                    WItemSelectionItemTyped< WLWindowFunction::WLEWindow > >()->getValue();

    if( m_useCuda->get() )
    {
#ifdef FOUND_CUDA
        infoLog() << "Using FIR filter for CUDA.";
        m_firFilter = WFIRFilter::SPtr(
                        new WFIRFilterCuda( fType, wType, m_order->get(), m_samplingFreq->get() * WLUnits::Hz,
                                        m_cFreq1->get() * WLUnits::Hz, m_cFreq2->get() * WLUnits::Hz ) );
#else
        errorLog() << "Build process has detected, that your machine has no CUDA support! Using CPU instead.";
        m_firFilter = WFIRFilter::SPtr(
                        new WFIRFilterCpu( fType, wType, m_order->get(), m_samplingFreq->get() * WLUnits::Hz,
                                        m_cFreq1->get() * WLUnits::Hz, m_cFreq2->get() * WLUnits::Hz ) );
#endif // FOUND_CUDA
    }
    else
    {
        infoLog() << "Using FIR filter for CPU.";
        m_firFilter = WFIRFilter::SPtr(
                        new WFIRFilterCpu( fType, wType, m_order->get(), m_samplingFreq->get() * WLUnits::Hz,
                                        m_cFreq1->get() * WLUnits::Hz, m_cFreq2->get() * WLUnits::Hz ) );
    }

    WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
    processReset( labp );

    progress->finish();
    m_progress->removeSubProgress( progress );
}

void WMFIRFilter::hdlDesignButtonPressed( void )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Updating FIR Filter" ) );
    m_progress->addSubProgress( progress );

    m_firFilter->setFilterType(
                    m_filterTypeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WFIRFilter::WEFilterType::Enum > >()->getValue() );
    m_firFilter->setWindowType(
                    m_windowSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WLWindowFunction::WLEWindow > >()->getValue() );
    m_firFilter->setOrder( m_order->get() );
    m_firFilter->setSamplingFrequency( m_samplingFreq->get() * WLUnits::Hz );
    m_firFilter->setCutOffFrequency1( m_cFreq1->get() * WLUnits::Hz );
    m_firFilter->setCutOffFrequency2( m_cFreq2->get() * WLUnits::Hz );
    m_firFilter->design();

    m_designTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    infoLog() << "New filter designed!";

    progress->finish();
    m_progress->removeSubProgress( progress );
}

void WMFIRFilter::cbFilterTypeChanged( void )
{
    debugLog() << __func__ << "() called!";
    if( ( WFIRFilter::WEFilterType::name( WFIRFilter::WEFilterType::BANDPASS ).compare(
                    m_filterTypeSelection->get().at( 0 )->getName() ) == 0
                    || WFIRFilter::WEFilterType::name( WFIRFilter::WEFilterType::BANDSTOP ).compare(
                                    m_filterTypeSelection->get().at( 0 )->getName() ) == 0 ) )
    {
        m_cFreq2->setHidden( false );
    }
    else
    {
        m_cFreq2->setHidden( true );
    }
}

bool WMFIRFilter::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMFIRFilter", __func__ );

    WLEMMeasurement::SPtr emmOut;

    // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
    debugLog() << "Data received ...";
    debugLog() << "EMM modalities: " << emmIn->getModalityCount();

    // Create output data
    emmOut.reset( new WLEMMeasurement( *emmIn ) );

    std::vector< WLEMData::SPtr > emdsIn = emmIn->getModalityList();
    for( std::vector< WLEMData::SPtr >::const_iterator emdIn = emdsIn.begin(); emdIn != emdsIn.end(); ++emdIn )
    {
        debugLog() << "EMD type: " << ( *emdIn )->getModalityType();
        if( ( *emdIn )->getSampFreq().value() != m_samplingFreq->get( false ) )
        {
            infoLog() << "Skip modality for FIR filter, because sampling frequencies are not equals.";
            emmOut->addModality( *emdIn );
            continue;
        }
#ifdef DEBUG
        // Show some input pieces
        const size_t nbChannels = ( *emdIn )->getNrChans();
        debugLog() << "EMD channels: " << nbChannels;
        const size_t nbSamlesPerChan = nbChannels > 0 ? ( *emdIn )->getSamplesPerChan() : 0;
        debugLog() << "EMD samples per channel: " << nbSamlesPerChan;
        debugLog() << "Input pieces:\n" << WLEMData::dataToString( ( *emdIn )->getData(), 5, 10 );
#endif // DEBUG
        try
        {
            WLEMData::SPtr emdOut = m_firFilter->filter( ( *emdIn ) );
            emmOut->addModality( emdOut );
#ifdef DEBUG
            // Show some filtered pieces
            debugLog() << "Filtered pieces:\n" << WLEMData::dataToString( emdOut->getData(), 5, 10 );
#endif // DEBUG
        }
        catch( const WException& e )
        {
            errorLog() << e.what();
            return false;
        }

    }
    m_firFilter->doPostProcessing( emmOut, emmIn );

    viewUpdate( emmOut );

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    labp->setEmm( emmOut );
    m_output->updateData( labp );

    return true;
}

bool WMFIRFilter::processInit( WLEMMCommand::SPtr cmdIn )
{
    bool rc = true;
    if( cmdIn->hasEmm() )
    {
        WLEMMeasurement::ConstSPtr emm = cmdIn->getEmm();
        WLEMData::ConstSPtr emd;

        WLFreqT samplFreqEeg = 0.0 * WLUnits::Hz;
        if( emm->hasModality( WLEModality::EEG ) )
        {
            emd = emm->getModality( WLEModality::EEG );
            samplFreqEeg = emd->getSampFreq();
        }

        WLFreqT samplFreqMeg = 0.0 * WLUnits::Hz;
        if( emm->hasModality( WLEModality::MEG ) )
        {
            emd = emm->getModality( WLEModality::MEG );
            samplFreqMeg = emd->getSampFreq();
        }

        WLFreqT samplFreq = 0.0 * WLUnits::Hz;
        if( samplFreqEeg == samplFreqMeg && samplFreqEeg > 0.0 * WLUnits::Hz )
        {
            samplFreq = samplFreqEeg;
        }
        else
            if( samplFreqEeg < 0.1 * WLUnits::Hz || samplFreqMeg < 0.1 * WLUnits::Hz )
            {
                samplFreq = std::max( samplFreqEeg.value(), samplFreqMeg.value() ) * WLUnits::Hz;
            }
        if( samplFreq > 0.0 * WLUnits::Hz )
        {
            infoLog() << "Init filter with new sampling rate: " << samplFreq;
            m_samplingFreq->set( samplFreq.value(), true );
            hdlDesignButtonPressed();
        }
        else
        {
            infoLog() << "No sampling rate to initialize!";
            rc = false;
        }
    }
    m_output->updateData( cmdIn );
    return rc;
}

bool WMFIRFilter::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_input->clear();
    viewReset();
    m_firFilter->reset();
    m_output->updateData( cmdIn );
    return true;
}

