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

#include <cstdlib> // srand()

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/module/WLConstantsModule.h"

#include "WMEMMGenerator.h"
#include "WMEMMGenerator.xpm"

W_LOADABLE_MODULE( WMEMMGenerator )

std::string WMEMMGenerator::EDataStatus::name( EDataStatus::Enum val )
{
    switch( val )
    {
        case NO_DATA:
            return "No data generated";
        case DATA_GENERATED:
            return "EMM data generated";
        case GENERATING_DATA:
            return "Generating data ...";
        case DATA_ERROR:
            return "Could not generate data!";
        default:
            return "Unknown status!";
    }
}

WMEMMGenerator::WMEMMGenerator()
{
    m_dataStatus = EDataStatus::NO_DATA;
}

WMEMMGenerator::~WMEMMGenerator()
{
}

const std::string WMEMMGenerator::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " EMM Generator";
}

const std::string WMEMMGenerator::getDescription() const
{
    return "Generates a random EMM object for testing purpose.";
}

WModule::SPtr WMEMMGenerator::factory() const
{
    return WModule::SPtr( new WMEMMGenerator );
}

const char** WMEMMGenerator::getXPMIcon() const
{
    return module_xpm;
}

void WMEMMGenerator::connectors()
{
    WModule::connectors();

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEMMGenerator::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propSamplFreq = m_properties->addProperty( "Sampling Frequency:", "Sampling Frequency in Hz.", 1000.0 );
    m_propSamplFreq->setMin( 1 );
    m_propSamplFreq->setMax( 9999 );

    m_propLength = m_properties->addProperty( "Length:", "Length of the measurement in seconds.", 60 );
    m_propLength->setMin( 1 );
    m_propLength->setMax( 3600 );

    m_propChans = m_properties->addProperty( "Channels:", "Number of channels to generate.", 60 );
    m_propChans->setMin( 1 );
    m_propChans->setMax( 400 );

    m_propDataStatus = m_properties->addProperty( "Status:", "Number of channels to generate.",
                    EDataStatus::name( m_dataStatus ) );
    m_propDataStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgGenerate = m_properties->addProperty( "Generate:", "Generate", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
}

void WMEMMGenerator::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();
}

void WMEMMGenerator::moduleMain()
{
    moduleInit();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
        }

        if( m_trgGenerate->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleTrgGenerate();
        }
    }
}

void WMEMMGenerator::handleTrgGenerate()
{
    debugLog() << "handleTrgGenerate() called!";

    updateDataStatus( EDataStatus::GENERATING_DATA );
    if( generateEMM() )
    {
        infoLog() << "EMM generated:\n" << *m_emm;
        updateDataStatus( EDataStatus::DATA_GENERATED );

        WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
        cmd->setEmm( m_emm );
        m_output->updateData( cmd );
    }
    else
    {
        errorLog() << "EMM could";
        updateDataStatus( EDataStatus::DATA_ERROR );
    }
    m_trgGenerate->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMEMMGenerator::generateEMM()
{
    m_emm.reset();

    const float sampl_freq = m_propSamplFreq->get();
    const size_t length = m_propLength->get();
    const size_t samples = sampl_freq * length;
    const size_t channels = m_propChans->get();

    WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
    WLEMDEEG::SPtr eeg( new WLEMDEEG() );

    eeg->setSampFreq( sampl_freq );
    eeg->getData().resize( channels, samples );
    for( size_t chan = 0; chan < channels; ++chan )
    {
        WLEMData::ChannelT channel( samples );
        for( size_t smp = 0; smp < samples; ++smp )
        {
            channel( smp ) = ( 30.0 * ( WLEMData::ScalarT )rand() / RAND_MAX - 15.0 );
        }
        WPosition::ValueType a = ( WPosition::ValueType )rand() / RAND_MAX - 0.5;
        WPosition::ValueType b = ( WPosition::ValueType )rand() / RAND_MAX - 0.5;
        WPosition::ValueType c = ( WPosition::ValueType )rand() / RAND_MAX - 0.5;
        WPosition::ValueType m = sqrt( a * a + b * b + c * c );
        WPosition::ValueType r = 100;
        a *= r / m;
        b *= r / m;
        c *= r / m;
        eeg->getChannelPositions3d()->push_back( WPosition( a, b, abs( c ) ) * 0.001 );
        eeg->getData().row( chan ) = channel;
    }

    emm->addModality( eeg );
    m_emm = emm;
    return true;
}
