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

#include <limits> // max double
#include <list>

#include <core/common/WPathHelper.h>
#include <core/kernel/WKernel.h>
#include <core/ui/WUIWidgetFactory.h>
#include <core/ui/WUI.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMDHPI.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "core/util/WLGeometry.h"

#include "WContinuousPositionEstimation.h"
#include "WMHeadPositionEstimation.xpm"
#include "WMHeadPositionEstimation.h"

#include "core/io/WLReaderMAT.h"

W_LOADABLE_MODULE( WMHeadPositionEstimation )

#define HPI_TEST 1

static const double HPI1_FREQ = 154.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 1 in Hz. */
static const double HPI2_FREQ = 158.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 2 in Hz. */
static const double HPI3_FREQ = 162.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 3 in Hz. */
static const double HPI4_FREQ = 166.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 4 in Hz. */
static const double HPI5_FREQ = 170.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 5 in Hz. */

static const double WINDOWS_SIZE = 200.0; /**< Default windows size in milliseconds. */

static const std::string STATUS_OK = "Ok"; /**< Indicates the module status is ok. */

WMHeadPositionEstimation::WMHeadPositionEstimation()
{
    m_lastParams.setZero();
}

WMHeadPositionEstimation::~WMHeadPositionEstimation()
{
}

const std::string WMHeadPositionEstimation::getName() const
{
    return WLConstantsModule::generateModuleName( "Head Position Estimation" );
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
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMHeadPositionEstimation::properties()
{
    WModule::properties();

    m_condition = WCondition::SPtr( new WCondition() );

    m_propGroupExtraction = m_properties->addPropertyGroup( "HPI Signal Extraction", "HPI Signal Extraction" );

    m_propHpi1Freq = m_propGroupExtraction->addProperty( "HPI #1 Frequency [Hz]:", "Frequency of HPI coil 1 in Hertz.",
                    HPI1_FREQ );
    m_propHpi2Freq = m_propGroupExtraction->addProperty( "HPI #2 Frequency [Hz]:", "Frequency of HPI coil 2 in Hertz.",
                    HPI2_FREQ );
    m_propHpi3Freq = m_propGroupExtraction->addProperty( "HPI #3 Frequency [Hz]:", "Frequency of HPI coil 3 in Hertz.",
                    HPI3_FREQ );
    m_propHpi4Freq = m_propGroupExtraction->addProperty( "HPI #4 Frequency [Hz]:", "Frequency of HPI coil 4 in Hertz.",
                    HPI4_FREQ );
    m_propHpi5Freq = m_propGroupExtraction->addProperty( "HPI #5 Frequency [Hz]:", "Frequency of HPI coil 5 in Hertz.",
                    HPI5_FREQ );

    m_propWindowsSize = m_propGroupExtraction->addProperty( "Windows Size [ms]:", "Windows size in milliseconds.", WINDOWS_SIZE );

    m_propStatus = m_propGroupExtraction->addProperty( "Status:", "Reports the status of actions.", STATUS_OK );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgApplySettings = m_propGroupExtraction->addProperty( "Apply Settings:", "Apply", WPVBaseTypes::PV_TRIGGER_READY,
                    m_condition );
    m_trgApplySettings->changed( true );

    m_propGroupEstimation = m_properties->addPropertyGroup( "Head Position Estimation", "Head Position Estimation" );

    m_propMaxIterations = m_propGroupEstimation->addProperty( "Max. Iterations:",
                    "Maximum iterations for minimization algorithm.", 500 );
    m_propEpsilon = m_propGroupEstimation->addProperty( "Epsilon:", "Epsilon/threshold for minimization algorithm.", 1e-4 );

    m_propInitFactor = m_propGroupEstimation->addProperty( "Initial Factor:", "Initial factor to create initial parameter set.",
                    2.0 );

    m_propInitRz = m_propGroupEstimation->addProperty( "Rz [°]:",
                    "Initial rotation around the z axis for z-y-x rotation degrees.", 0.0 );
    m_propInitRy = m_propGroupEstimation->addProperty( "Ry [°]:",
                    "Initial rotation around the y axis for z-y-x rotation degrees.", 0.0 );
    m_propInitRx = m_propGroupEstimation->addProperty( "Rx [°]:",
                    "Initial rotation around the x axis for z-y-x rotation degrees.", 0.0 );

    m_propInitTx = m_propGroupEstimation->addProperty( "Tx [m]:", "Initial x translation in meter.", 0.0 );
    m_propInitTy = m_propGroupEstimation->addProperty( "Ty [m]:", "Initial y translation in meter.", 0.0 );
    m_propInitTz = m_propGroupEstimation->addProperty( "Tz [m]:", "Initial z translation in meter.", 0.0 );

    m_propErrorMin = m_infoProperties->addProperty( "Error (min.):", "Min. fitting error of last block.", 0.0 );
    m_propErrorMin->setPurpose( PV_PURPOSE_INFORMATION );

    m_propErrorAvg = m_infoProperties->addProperty( "Error (avg.):", "Average fitting error of last block.", 0.0 );
    m_propErrorAvg->setPurpose( PV_PURPOSE_INFORMATION );

    m_propErrorMax = m_infoProperties->addProperty( "Error (max):", "Max. fitting error of last block.", 0.0 );
    m_propErrorMax->setPurpose( PV_PURPOSE_INFORMATION );

    m_propItMin = m_infoProperties->addProperty( "Iterations (min.):", "Min. iterations of last block.", 0 );
    m_propItMin->setPurpose( PV_PURPOSE_INFORMATION );

    m_propItAvg = m_infoProperties->addProperty( "Iterations (avg.):", "Average iterations of last block.", 0.0 );
    m_propItAvg->setPurpose( PV_PURPOSE_INFORMATION );

    m_propItMax = m_infoProperties->addProperty( "Iterations (max):", "Max. iterations of last block.", 0 );
    m_propItMax->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMHeadPositionEstimation::viewInit()
{
    // Create main widget
    WUIWidgetFactory::SPtr factory = WKernel::getRunningKernel()->getUI()->getWidgetFactory();
    m_widget = factory->createGridWidget( getName() );

    // Create views for top, side, front
    m_widgetTop = factory->createViewWidget( "Top View", WGECamera::ORTHOGRAPHIC, m_shutdownFlag.getValueChangeCondition(),
                    m_widget );
    m_widget->placeWidget( m_widgetTop, 0, 0 );
    m_drawableTop = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetTop ) );
    m_drawableTop->setView( WLEMDDrawable3DHPI::VIEW_TOP );

    m_widgetSide = factory->createViewWidget( "Side View", WGECamera::ORTHOGRAPHIC, m_shutdownFlag.getValueChangeCondition(),
                    m_widget );
    m_widget->placeWidget( m_widgetSide, 1, 0 );
    m_drawableSide = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetSide ) );
    m_drawableSide->setView( WLEMDDrawable3DHPI::VIEW_SIDE );

    m_widgetFront = factory->createViewWidget( "Front View", WGECamera::ORTHOGRAPHIC, m_shutdownFlag.getValueChangeCondition(),
                    m_widget );
    m_widget->placeWidget( m_widgetFront, 2, 0 );
    m_drawableFront = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetFront ) );
    m_drawableFront->setView( WLEMDDrawable3DHPI::VIEW_FRONT );
}

void WMHeadPositionEstimation::viewUpdate( WLEMMeasurement::SPtr emm )
{
    if( m_widget->isClosed() || !m_widget->isVisible() )
    {
        return;
    }
    m_drawableTop->draw( emm );
    m_drawableSide->draw( emm );
    m_drawableFront->draw( emm );
}

void WMHeadPositionEstimation::viewReset()
{
    m_drawableTop = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetTop ) );
    m_drawableTop->setView( WLEMDDrawable3DHPI::VIEW_TOP );

    m_drawableSide = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetSide ) );
    m_drawableSide->setView( WLEMDDrawable3DHPI::VIEW_SIDE );

    m_drawableFront = WLEMDDrawable3DHPI::SPtr( new WLEMDDrawable3DHPI( m_widgetFront ) );
    m_drawableFront->setView( WLEMDDrawable3DHPI::VIEW_FRONT );
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

    viewInit();

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

        if( m_trgApplySettings->changed( true ) )
        {
            hdlApplyFreq();
            m_trgApplySettings->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
            process( cmdIn );
        }
    }
}

bool WMHeadPositionEstimation::hdlApplyFreq()
{
    debugLog() << __func__ << "() called!";

    m_hpiSignalExtraction.reset( new WHPISignalExtraction() );

    m_hpiSignalExtraction->addFrequency( m_propHpi1Freq->get() * WLUnits::Hz );
    m_hpiSignalExtraction->addFrequency( m_propHpi2Freq->get() * WLUnits::Hz );
    m_hpiSignalExtraction->addFrequency( m_propHpi3Freq->get() * WLUnits::Hz );
    m_hpiSignalExtraction->addFrequency( m_propHpi4Freq->get() * WLUnits::Hz );
    m_hpiSignalExtraction->addFrequency( m_propHpi5Freq->get() * WLUnits::Hz );

    const WLTimeT win_size = m_hpiSignalExtraction->setWindowsSize( m_propWindowsSize->get() * 1e-3 * WLUnits::s );
    m_hpiSignalExtraction->setStepSize( win_size );
    m_propWindowsSize->set( win_size.value() * 1e3, true );

    infoLog() << *m_hpiSignalExtraction;
    m_propStatus->set( STATUS_OK, true );
    return true;
}

bool WMHeadPositionEstimation::setHpiCoilFreqs( const WLEMMHpiInfo& hpiInfo )
{
    debugLog() << __func__ << "() called!";
    if( hpiInfo.getHpiFrequencies().empty() )
    {
        errorLog() << "No HPI coil frequencies available!";
        return false;
    }

    WLEMMHpiInfo::HpiFrequenciesT freqs = hpiInfo.getHpiFrequencies();
    if( freqs.size() != 5 )
    {
        errorLog() << "5 HPI coils are needed, at the moment!";
        return false;
    }

    WLEMMHpiInfo::HpiFrequenciesT::iterator it = freqs.begin();
    m_propHpi1Freq->set( it->value(), true );
    ++it;
    m_propHpi2Freq->set( it->value(), true );
    ++it;
    m_propHpi3Freq->set( it->value(), true );
    ++it;
    m_propHpi4Freq->set( it->value(), true );
    ++it;
    m_propHpi5Freq->set( it->value(), true );

    return hdlApplyFreq();
}

bool WMHeadPositionEstimation::processInit( WLEMMCommand::SPtr cmdIn )
{
    debugLog() << __func__ << "() called!";

    bool rc = true;
    if( cmdIn->hasEmm() )
    {
        infoLog() << "Initializing HPI coil frequencies.";
        WLEMMeasurement::ConstSPtr emm = cmdIn->getEmm();
        rc &= setHpiCoilFreqs( *emm->getHpiInfo() );
    }
    else
    {
        rc &= hdlApplyFreq();
    }

    m_output->updateData( cmdIn );
    return rc;
}

bool WMHeadPositionEstimation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", __func__ );

    WLEMDMEG::SPtr magIn;
    if( !extractMagnetometer( magIn, emmIn ) )
    {
        return false;
    }

    if( !m_hpiSignalExtraction )
    {
        infoLog() << "Initializing HPI coil frequencies.";
        if( !setHpiCoilFreqs( *emmIn->getHpiInfo() ) )
        {
            infoLog() << "Using default or user defined HPI coil frequencies!";
            if( !hdlApplyFreq() )
            {
                errorLog() << "Could not create HPI signal extraction!";
                return false;
            }
        }
    }

    WLEMDHPI::SPtr hpiOut;
    if( !extractHpiSignals( hpiOut, magIn ) )
    {
        WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmdOut->setEmm( emmIn );
        m_output->updateData( cmdOut );
        return false;
    }

    if( !hpiOut->setChannelPositions3d( emmIn->getDigPoints() ) )
    {
        errorLog() << "Could not set isotrak positions for HPI coils!";
        return false;
    }

    if( !estimateHeadPosition( hpiOut, magIn ) )
    {
        WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmdOut->setEmm( emmIn );
        m_output->updateData( cmdOut );
        return false;
    }

    // Reconstructed HPI amplitudes and positions
    WLEMMeasurement::SPtr emmOut = emmIn->clone(); // TODO(pieloth): Do we really need to clone this?
    emmOut->setModalityList( emmIn->getModalityList() );
    emmOut->addModality( hpiOut );
    viewUpdate( emmOut );

    WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmdOut->setEmm( emmOut );
    m_output->updateData( cmdOut );

    return true;
}

bool WMHeadPositionEstimation::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_hpiSignalExtraction.reset();
    m_optim.reset();
    m_lastParams.setZero();
    m_output->updateData( cmdIn );
    m_propErrorAvg->set( 0.0, true );
    viewReset();
    return true;
}

bool WMHeadPositionEstimation::processTime( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

bool WMHeadPositionEstimation::processMisc( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

bool WMHeadPositionEstimation::extractMagnetometer( WLEMDMEG::SPtr& magOut, WLEMMeasurement::ConstSPtr emmIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", __func__ );
    if( !emmIn->hasModality( WLEModality::MEG ) )
    {
        errorLog() << "No MEG data!";
        return false;
    }

    WLEMDMEG::ConstSPtr megIn = emmIn->getModality< const WLEMDMEG >( WLEModality::MEG );
    if( WLEMDMEG::extractCoilModality( magOut, megIn, WLEModality::MEG_MAG, false ) )
    {
        return true;
    }
    else
    {
        errorLog() << "Could not extract magnetometer!";
        return false;
    }
}

bool WMHeadPositionEstimation::extractHpiSignals( WLEMDHPI::SPtr& hpiOut, WLEMDMEG::ConstSPtr magIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", __func__ );

    if( magIn->getSampFreq() != m_hpiSignalExtraction->getSamplingFrequency() )
    {
        m_hpiSignalExtraction->setSamplingFrequency( magIn->getSampFreq() );
        infoLog() << "Update signal extraction sfreq: " << m_hpiSignalExtraction->getSamplingFrequency();
    }

    const bool rc = m_hpiSignalExtraction->reconstructAmplitudes( hpiOut, magIn );
    if( rc )
    {
        return true;
    }
    else
    {
        errorLog() << "reconstructAmplitudes() error!";
        return false;
    }

    return true;
}

bool WMHeadPositionEstimation::estimateHeadPosition( WLEMDHPI::SPtr hpiInOut, WLEMDMEG::ConstSPtr magMag )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", __func__ );

    // Prepare optimization and MEG magnetometer data
    if( !m_optim )
    {
        WLEMDMEG::PositionsT::ConstSPtr magPos = magMag->getChannelPositions3d();
        WLArrayList< WVector3f >::ConstSPtr magOri = magMag->getEz();
        debugLog() << "magPos: " << *magPos;
        debugLog() << "magOri: " << *magOri;

        m_optim.reset( new WContinuousPositionEstimation( *hpiInOut->getChannelPositions3d(), *magPos, *magOri ) );
        m_optim->setMaximumIterations( m_propMaxIterations->get() );
        m_optim->setEpsilon( m_propEpsilon->get() );
        m_optim->setInitialFactor( m_propInitFactor->get() );
        const double toRadFac = M_PI / 180;
        m_lastParams( 0 ) = m_propInitRx->get() * toRadFac;
        m_lastParams( 1 ) = m_propInitRy->get() * toRadFac;
        m_lastParams( 2 ) = m_propInitRz->get() * toRadFac;
        m_lastParams( 3 ) = m_propInitTx->get();
        m_lastParams( 4 ) = m_propInitTy->get();
        m_lastParams( 5 ) = m_propInitTz->get();
        infoLog() << *m_optim;
    }

    m_optim->setMaximumIterations( m_propMaxIterations->get() );
    m_optim->setEpsilon( m_propEpsilon->get() );
    m_optim->setInitialFactor( m_propInitFactor->get() );

    // Estimate positions
    m_optim->setData( hpiInOut->getData() );
    WContinuousPositionEstimation::MatrixT::Index smp;
    const WContinuousPositionEstimation::MatrixT::Index n_smp = hpiInOut->getData().cols();
    WLArrayList< WLEMDHPI::TransformationT >::SPtr trans = WLArrayList< WLEMDHPI::TransformationT >::instance();
    trans->reserve( n_smp );

    double errorMin = std::numeric_limits< double >::max();
    double errorAvg = 0.0;
    double errorMax = std::numeric_limits< double >::min();

    size_t itMin = std::numeric_limits< size_t >::max();
    double itAvg = 0.0;
    size_t itMax = std::numeric_limits< size_t >::min();

    for( smp = 0; smp < n_smp; ++smp )
    {
        // Do optimization
        const WContinuousPositionEstimation::Converged conv = m_optim->optimize( m_lastParams );
        m_lastParams = m_optim->getResultParams();
        const double error = m_optim->getResultError();
        const size_t iterations = m_optim->getResultIterations();
        WLTransformation::ConstSPtr result = m_optim->getResultTransformation();

        // Calculate errors for informational output
        if( error < errorMin )
        {
            errorMin = error;
        }
        if( error > errorMax )
        {
            errorMax = error;
        }
        errorAvg += error;

        // Calculate itreations for informational output
        if( iterations < itMin )
        {
            itMin = iterations;
        }
        if( iterations > itMax )
        {
            itMax = iterations;
        }
        itAvg += iterations;

        // Store result and set next sample
        trans->push_back( *result );
        m_optim->nextSample();

#ifdef HPI_TEST
        // Debug output
        debugLog() << ">>>>> BEGIN";
        debugLog() << "Estimation: " << conv << " " << iterations << " " << error;
        debugLog() << "Transformation:\n" << result->data();

        WLPositions::SPtr pos = *result * *hpiInOut->getChannelPositions3d();
        debugLog() << "HPI positions:\n";
        for( WLPositions::IndexT i = 0; i < pos->size(); ++i )
        {
            debugLog() << pos->at( i ).x() << " " << pos->at( i ).y() << " " << pos->at( i ).z();
        }
        debugLog() << "<<<<< END";
#endif // HPI_TEST
    }
    hpiInOut->setTransformations( trans );

    m_propErrorMin->set( errorMin, true );
    m_propErrorAvg->set( errorAvg / n_smp, true );
    m_propErrorMax->set( errorMax, true );

    m_propItMin->set( itMin, true );
    m_propItAvg->set( itAvg / n_smp, true );
    m_propItMax->set( itMax, true );

    debugLog() << "Error (average): " << errorAvg / n_smp;
    debugLog() << "Iterations (average): " << itAvg / n_smp;

    return true;
}
