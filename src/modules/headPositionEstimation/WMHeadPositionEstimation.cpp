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

#include <list>

#include <core/common/WPathHelper.h>

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

static const double HPI1_FREQ = 154.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 1 in Hz. */
static const double HPI2_FREQ = 158.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 2 in Hz. */
static const double HPI3_FREQ = 162.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 3 in Hz. */
static const double HPI4_FREQ = 166.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 4 in Hz. */
static const double HPI5_FREQ = 170.0; /**< Default frequency (sfreq < 600Hz) for HPI coil 5 in Hz. */

static const double WINDOWS_SIZE = 200.0; /**< Default windows size in milliseconds. */
static const double STEP_SIZE = 10.0; /**< Default step size in milliseconds. */

static const std::string STATUS_OK = "Ok"; /**< Indicates the module status is ok. */
static const std::string STATUS_ERROR = "Error"; /**< Indicates an error in module. */

WMHeadPositionEstimation::WMHeadPositionEstimation()
{
    m_lastParams.setZero();
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
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMHeadPositionEstimation::properties()
{
    WLModuleDrawable::properties();
    hideComputeModalitySelection( true );
    std::list< WLEModality::Enum > viewMods;
    viewMods.push_back( WLEModality::HPI );
    viewMods.push_back( WLEModality::MEG );
    viewMods.push_back( WLEModality::MEG_GRAD );
    viewMods.push_back( WLEModality::MEG_GRAD_MERGED );
    viewMods.push_back( WLEModality::MEG_MAG );
    setViewModalitySelection( viewMods );

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
    m_propStepSize = m_propGroupExtraction->addProperty( "Step Size [ms]:", "Step size in milliseconds.", STEP_SIZE );

    m_propStatus = m_propGroupExtraction->addProperty( "Status:", "Reports the status of actions.", STATUS_OK );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgApplySettings = m_propGroupExtraction->addProperty( "Apply Settings:", "Apply", WPVBaseTypes::PV_TRIGGER_READY,
                    m_condition );

    m_propGroupEstimation = m_properties->addPropertyGroup( "Head Position Estimation", "Head Position Estimation" );

    m_propMaxIterations = m_propGroupEstimation->addProperty( "Max. Iterations:",
                    "Maximum iterations for minimization algorithm.", 128 );
    m_propEpsilon = m_propGroupEstimation->addProperty( "Epsilon:", "Epsilon/threshold for minimization algorithm.", 1.0e-6 );

    m_propInitAlpha = m_propGroupEstimation->addProperty( "Rz:", "Initial step: alpha angle in degrees for z-y-z rotation.",
                    10.0 );
    m_propInitBeta = m_propGroupEstimation->addProperty( "Ry':", "Initial step: beta angle in degrees for z-y-z rotation.",
                    10.0 );
    m_propInitGamma = m_propGroupEstimation->addProperty( "Rz'':", "Initial step: gamma angle in degrees for z-y-z rotation.",
                    10.0 );

    m_propInitX = m_propGroupEstimation->addProperty( "Tx:", "Initial step: x translation in meter.", 0.01 );
    m_propInitY = m_propGroupEstimation->addProperty( "Ty:", "Initial step: y translation in meter.", 0.01 );
    m_propInitZ = m_propGroupEstimation->addProperty( "Tz:", "Initial step: z translation in meter.", 0.01 );
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

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );
    handleApplyFreq();

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
    debugLog() << "handleApplyFreq() called!";

    m_hpiSignalExtraction.reset( new WHPISignalExtraction() );
    m_hpiSignalExtraction->setWindowsSize( m_propWindowsSize->get() );
    m_hpiSignalExtraction->setStepSize( m_propStepSize->get() );

    m_hpiSignalExtraction->addFrequency( m_propHpi1Freq->get() );
    m_hpiSignalExtraction->addFrequency( m_propHpi2Freq->get() );
    m_hpiSignalExtraction->addFrequency( m_propHpi3Freq->get() );
    m_hpiSignalExtraction->addFrequency( m_propHpi4Freq->get() );
    m_hpiSignalExtraction->addFrequency( m_propHpi5Freq->get() );

    m_propStatus->set( STATUS_OK, true );
    return true;
}

bool WMHeadPositionEstimation::processInit( WLEMMCommand::SPtr cmdIn )
{
    const bool rc = handleApplyFreq();
    m_output->updateData( cmdIn );
    return rc;
}

bool WMHeadPositionEstimation::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", "processCompute" );
    WLEMDHPI::SPtr hpiOut;
    // TODO(pieloth): states over 2 methods with m_lastEmm is bad. change it!
    if( !extractHpiSignals( hpiOut, emmIn ) )
    {
        WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmdOut->setEmm( emmIn );
        m_output->updateData( cmdOut );
        return false;
    }

    if( !m_lastEmm )
    {
        m_lastEmm = emmIn;
        return false;
    }

    // TODO(pieloth): Requires origin positions, isotrak and estimates positions.
    if( !hpiOut->setChannelPositions3d( emmIn->getDigPoints() ) )
    {
        warnLog() << "Could not set isotrak positions for HPI coils!";
    }

    if( !estimateHeadPosition( 0, hpiOut, emmIn ) )
    {
        WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmdOut->setEmm( emmIn );
        m_output->updateData( cmdOut );
        return false;
    }

    // Reconstructed HPI amplitudes and positions are for previous EMM packet
    if( m_lastEmm.get() != NULL )
    {
        WLEMMeasurement::SPtr emmOut = m_lastEmm->clone();
        emmOut->setModalityList( m_lastEmm->getModalityList() );
        emmOut->addModality( hpiOut );
        // TODO add/set positions/transformation
        viewUpdate( emmOut );

        WLEMMCommand::SPtr cmdOut( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
        cmdOut->setEmm( emmOut );
        m_output->updateData( cmdOut );
        m_lastEmm = emmIn;
        return true;
    }
    else
    {
        errorLog() << "Previous EMM packet is emtpy!";
        return false;
    }
}

bool WMHeadPositionEstimation::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_lastEmm.reset();
    m_hpiSignalExtraction->reset();
    m_optim.reset();
    m_output->updateData( cmdIn );
    return true;
}

bool WMHeadPositionEstimation::extractHpiSignals( WLEMDHPI::SPtr& hpiOut, WLEMMeasurement::ConstSPtr emmIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", "extractHpiSignals" );

    if( !emmIn->hasModality( WLEModality::MEG ) )
    {
        errorLog() << "No MEG data!";
        return false;
    }

    WLEMDMEG::ConstSPtr megIn = emmIn->getModality< const WLEMDMEG >( WLEModality::MEG );
    if( megIn->getSampFreq() != m_hpiSignalExtraction->getSamplingFrequency() )
    {
        m_hpiSignalExtraction->setSamplingFrequency( megIn->getSampFreq() );
    }

    WLEMDMEG::SPtr megMag;
    if( !WLEMDMEG::extractCoilModality( megMag, megIn, WLEModality::MEG_MAG, true ) )
    {
        errorLog() << "Could not extract magnetometer!";
        return false;
    }

    const bool rc = m_hpiSignalExtraction->reconstructAmplitudes( hpiOut, megMag );
    if( rc )
    {
        return true;
    }
    else
        if( !rc && !m_lastEmm )
        {
            warnLog() << "reconstructAmplitudes() error, but first packet!";
            return true;
        }
        else
        {
            errorLog() << "reconstructAmplitudes() error!";
            m_lastEmm.reset();
            return false;
        }
    // Test code for matlab
//    WLReaderMAT::SPtr reader;
//    std::string fName = "/home/pieloth/hpi_data.mat";
//    try
//    {
//        reader.reset( new WLReaderMAT( fName ) );
//    }
//    catch( const WDHNoSuchFile& e )
//    {
//        errorLog() << "File does not exist: " << fName;
//        return false;
//    }
//
//    WLIOStatus::ioStatus_t status;
//    status = reader->init();
//    if( status != WLIOStatus::SUCCESS )
//    {
//        errorLog() << reader->getIOStatusDescription( status );
//        return false;
//    }
//
//    WLMatrix::SPtr mat;
//    status = reader->readMatrix( mat );
//    if( status != WLIOStatus::SUCCESS )
//    {
//        errorLog() << reader->getIOStatusDescription( status );
//        return false;
//    }
//    hpiOut.reset( new WLEMDHPI );
//    hpiOut->setData( mat );
//    hpiOut->setNrHpiCoils( 5 );
//    hpiOut->setSampFreq( 1.0 );
    return true;
}

bool WMHeadPositionEstimation::estimateHeadPosition( int out, WLEMDHPI::ConstSPtr hpiIn, WLEMMeasurement::ConstSPtr emmIn )
{
    WLTimeProfiler tp( "WMHeadPositionEstimation", "estimateHeadPosition" );

    // TODO
    if( !emmIn->hasModality( WLEModality::MEG ) )
    {
        errorLog() << "No MEG data!";
        return false;
    }

    if( !m_optim )
    {
        // Prepare MEG magnetometer data
        WLEMDMEG::ConstSPtr megIn = emmIn->getModality< const WLEMDMEG >( WLEModality::MEG );
        WLEMDMEG::SPtr megMag( new WLEMDMEG );
        WLEMDMEG::extractCoilModality( megMag, megIn, WLEModality::MEG_MAG, false );
        debugLog() << "meg_mag: " << *megMag;

        WLArrayList< WPosition >::ConstSPtr magPos = megMag->getChannelPositions3d();
        WLArrayList< WVector3f >::ConstSPtr magOri = megMag->getEz();
        debugLog() << "magPos: " << *magPos;
        debugLog() << "magOri: " << *magOri;

        m_optim.reset( new WContinuousPositionEstimation( *hpiIn->getChannelPositions3d(), *magPos, *magOri ) );
        m_optim->setMaximumIterations( m_propMaxIterations->get() );
        m_optim->setEpsilon( m_propEpsilon->get() );
        WContinuousPositionEstimation::ParamsT step = WContinuousPositionEstimation::ParamsT::Zero();
        const double toRadFac = M_PI / 180;
        step( 0 ) = m_propInitAlpha->get() * toRadFac;
        step( 1 ) = m_propInitBeta->get() * toRadFac;
        step( 2 ) = m_propInitGamma->get() * toRadFac;
        step( 3 ) = m_propInitX->get();
        step( 4 ) = m_propInitY->get();
        step( 5 ) = m_propInitZ->get();
        m_optim->setInitialStep( step );
        m_optim->setInitialFactor( 0.5 );
        debugLog() << *m_optim;
    }

    m_optim->setData( hpiIn->getData() );
    WContinuousPositionEstimation::MatrixT::Index smp;
    const WContinuousPositionEstimation::MatrixT::Index n_smp = hpiIn->getData().cols();
    for( smp = 0; smp < n_smp; ++smp )
    {
        m_optim->optimize( m_lastParams );
        m_lastParams = m_optim->getResultParams();
        debugLog() << "Estimation: " << m_optim->converged() << " " << m_optim->getResultIterations() << " "
                        << m_optim->getResultError();
        debugLog() << "Best:\n" << m_lastParams;
        debugLog() << "Transformation:\n" << m_optim->getResultTransformation();
        std::vector< WPosition > hpiPosNew = m_optim->getResultPositions();
        debugLog() << "HPI Positions:\n" << hpiPosNew[0] << "\n" << hpiPosNew[1] << "\n" << hpiPosNew[2] << "\n" << hpiPosNew[3]
                        << "\n" << hpiPosNew[4];
        m_optim->nextSample();
        // TODO(pieloth): store result
    }

    return false;
}
