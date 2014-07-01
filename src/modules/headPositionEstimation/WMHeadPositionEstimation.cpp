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

#include "WContinuousPositionEstimation.h"
#include "WMHeadPositionEstimation.xpm"
#include "WMHeadPositionEstimation.h"

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
                    "Maximum iterations for minimization algorithm.", 200 );
    m_propEpsilon = m_propGroupEstimation->addProperty( "Epsilon:", "Epsilon/threshold for minimization algorithm.", 1.0 );
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
    else
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

    // TODO(pieloth): Save magnetometer information, because they are not changing
    // Prepare MEG magnetometer data
    WLEMDMEG::ConstSPtr megIn = emmIn->getModality< const WLEMDMEG >( WLEModality::MEG );
    WLEMDMEG::SPtr megMag( new WLEMDMEG );
    WLEMDMEG::extractCoilModality( megMag, megIn, WLEModality::MEG_MAG, false );
    debugLog() << "meg_mag: " << *megMag;

    WLArrayList< WPosition >::ConstSPtr magPos = megMag->getChannelPositions3d();
    WLArrayList< WVector3f >::ConstSPtr magOri = megMag->getEz();
    debugLog() << "magPos: " << *magPos;
    debugLog() << "magOri: " << *magOri;

    std::vector< WContinuousPositionEstimation::PositionT > magPos4hpi( magPos->begin(), magPos->end() );
    std::vector< WContinuousPositionEstimation::Vector3T > magOri4hpi;
    magOri4hpi.reserve( magOri->size() );
    for( size_t i = 0; i < megMag->getNrChans(); ++i )
    {
        const WVector3f ori = magOri->at( i );
        magOri4hpi.push_back( WContinuousPositionEstimation::Vector3T( ori.x(), ori.y(), ori.z() ) );
    }
    debugLog() << "magPos4hpi: " << magPos4hpi.size();
    debugLog() << "magOri4hpi: " << magOri4hpi.size();

    WContinuousPositionEstimation est( magPos4hpi, magOri4hpi );
    est.setData( hpiIn->getData() );
    WContinuousPositionEstimation::PointT initial;
    WLArrayList< WPosition >::const_iterator it = hpiIn->getChannelPositions3d()->begin();
    int i = 0;
    for( ; it != hpiIn->getChannelPositions3d()->end(); ++it )
    {
        initial( 0, i + 0 ) = it->x();
        initial( 0, i + 1 ) = it->y();
        initial( 0, i + 2 ) = it->z();
        i += 3;
    }

    est.optimize( initial );
    //
    // Collect results
    const WContinuousPositionEstimation::PointT best = est.getBestVariable();
    // TODO(pieloth): remove logging
    debugLog() << "Estimation: " << est.isConverged() << " " << est.getIterations() << " " << est.func( best );
//    debugLog() << "hpiPos head: " << hpiPos->at( ch );
    debugLog() << "hpiPos device: " << best;

//    // TODO(pieloth)
//    // Get HPI data
//    const WLEMData::DataT& hpiData = hpiIn->getData();
//    debugLog() << "hpiData: " << hpiData.rows() << "x" << hpiData.cols();
//    WLArrayList< WPosition >::ConstSPtr hpiPos = hpiIn->getChannelPositions3d();
//    debugLog() << "hpiPos: " << *hpiPos;
//    for( size_t ch = 0; ch < hpiIn->getNrHpiCoils(); ++ch )
//    {
//        debugLog() << "HPI coil #" << ch;
//        // TODO(pieloth)
//        // Get data for HPI channel
//        WContinuousPositionEstimation::MatrixT hpiData4Coil( hpiData.rows() / hpiIn->getNrHpiCoils(), hpiData.cols() );
//        WLEMData::DataT::Index i = ch;
//        WContinuousPositionEstimation::MatrixT::Index j = 0;
//        while( i < hpiData.rows() )
//        {
//            hpiData4Coil.row( j ) = hpiData.row( i );
//            i += hpiIn->getNrHpiCoils();
//            ++j;
//        }
//        debugLog() << "hpiData4Coil: " << hpiData4Coil.rows() << "x" << hpiData4Coil.cols();
//
//        // Do estimation
//        WContinuousPositionEstimation est;
//        est.setMaximumIterations( 256 );
//        est.setEpsilon( 0.04 );
//        est.m_data = hpiData4Coil;
//        est.m_sensPos = magPos4hpi;
//        est.m_sensOri = magOri4hpi;
//        const WPosition hpiCoilPos = hpiPos->at( ch );
//        const WContinuousPositionEstimation::PointT p( hpiCoilPos.x(), hpiCoilPos.y(), hpiCoilPos.z() - 0.005 );
//        est.optimize( p );
//
//        // Collect results
//        const WContinuousPositionEstimation::PointT best = est.getBestVariable();
//        // TODO(pieloth): remove logging
//        debugLog() << "Estimation: " << est.isConverged() << " " << est.getIterations() << " " << est.func( best );
//        debugLog() << "hpiPos head: " << hpiPos->at( ch );
//        debugLog() << "hpiPos device: " << best;
//    }

    return false;
}
