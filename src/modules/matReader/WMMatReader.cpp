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

#include <core/common/WException.h>
#include <core/common/WPathHelper.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderMAT.h"
#include "core/io/WLReaderSourceSpace.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "reader/WReaderEEGPositions.h"

#include "WMMatReader.xpm"
#include "WMMatReader.h"

using namespace LaBP;

W_LOADABLE_MODULE( WMMatReader )

const double WMMatReader::SAMPLING_FEQUENCY = 1000;

const std::string WMMatReader::NONE = "none";
const std::string WMMatReader::SUCCESS_READ = "File successfully read.";
const std::string WMMatReader::ERROR_READ = "Could not read file!";

const std::string WMMatReader::ERROR_EMM = "Could not generate EMM object!";
const std::string WMMatReader::SUCCESS_EMM = "EMM object successfully created.";
const std::string WMMatReader::GENERATE_EMM = "Generating EMM object ...";

const std::string WMMatReader::READING_MAT = "Reading MAT-File ...";
const std::string WMMatReader::READING_LF = "Reading Leadfield ...";
const std::string WMMatReader::READING_SRC = "Reading Source Space ...";
const std::string WMMatReader::READING_SENSORS = "Reading Sensor positions ...";

WMMatReader::WMMatReader()
{
}

WMMatReader::~WMMatReader()
{
}

const std::string WMMatReader::getName() const
{
    return "MAT-File Reader";
}

const std::string WMMatReader::getDescription() const
{
    return "Reads a MATLAB MAT-file into EMM structure.";
}

WModule::SPtr WMMatReader::factory() const
{
    return WModule::SPtr( new WMMatReader );
}

const char** WMMatReader::getXPMIcon() const
{
    return module_xpm;
}

void WMMatReader::connectors()
{
    WModule::connectors();
    m_output.reset(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMMatReader::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propMatFile = m_properties->addProperty( "MAT-File:", "MATLAB MAT-File to read.", WPathHelper::getHomePath(),
                    m_propCondition );
    m_propMatFile->changed( true );

    m_propSensorFile = m_properties->addProperty( "Sensor Positions:", "FIFF file containing sensor positions.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_propSensorFile->changed( true );

    m_propLfFile = m_properties->addProperty( "Leadfield:", "FIFF file containing a Leadfield.", WPathHelper::getHomePath(),
                    m_propCondition );
    m_propLfFile->changed( true );

    m_propSrcSpaceFile = m_properties->addProperty( "Source Space:", "FIFF file containing a Source Space.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_propSrcSpaceFile->changed( true );

    m_propSamplFreq = m_properties->addProperty( "Sampling Frequency:", "Sampling Frequency of the data", SAMPLING_FEQUENCY,
                    m_propCondition );

    m_trgGenerate = m_properties->addProperty( "Generate EMM:", "Generate", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_status = m_properties->addProperty( "Status:", "Status", NONE );
    m_status->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMMatReader::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();
}

void WMMatReader::moduleMain()
{
    moduleInit();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
        }

        if( m_propMatFile->changed( true ) )
        {
            m_status->set( READING_MAT, true );
            if( handleMatFileChanged() )
            {
                m_status->set( SUCCESS_READ, true );
            }
            else
            {
                m_status->set( ERROR_READ, true );
            }
        }

        if( m_propSensorFile->changed( true ) )
        {
            m_status->set( READING_SENSORS, true );
            if( handleSensorFileChanged() )
            {
                m_status->set( SUCCESS_READ, true );
            }
            else
            {
                m_status->set( ERROR_READ, true );
            }
        }

        if( m_propLfFile->changed( true ) )
        {
            m_status->set( READING_LF, true );
            if( handleLfFileChanged() )
            {
                m_status->set( SUCCESS_READ, true );
            }
            else
            {
                m_status->set( ERROR_READ, true );
            }
        }

        if( m_propSrcSpaceFile->changed( true ) )
        {
            m_status->set( READING_SRC, true );
            if( handleSurfaceFileChanged() )
            {
                m_status->set( SUCCESS_READ, true );
            }
            else
            {
                m_status->set( ERROR_READ, true );
            }
        }

        if( m_trgGenerate->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            m_status->set( GENERATE_EMM, true );
            if( handleGenerateEMM() )
            {
                m_status->set( SUCCESS_EMM, true );
            }
            else
            {
                m_status->set( ERROR_EMM, true );
            }
        }
    }
}

bool WMMatReader::handleSensorFileChanged()
{
    const std::string fName = m_propSensorFile->get().string();
    infoLog() << "Start reading file: " << fName;
    m_sensorPos.reset();

    WReaderEEGPositions::SPtr reader;
    try
    {
        reader.reset( new WReaderEEGPositions( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    if( reader->read( m_sensorPos ) != WLIOStatus::SUCCESS )
    {
        errorLog() << ERROR_READ << " (Sensor Positions)";
        return false;
    }

    infoLog() << SUCCESS_READ << " Sensor Positions: " << m_sensorPos->size();
    return true;
}

bool WMMatReader::handleMatFileChanged()
{
    const std::string fName = m_propMatFile->get().string();
    infoLog() << "Start reading file: " << fName;

    WLReaderMAT::SPtr reader;
    try
    {
        reader.reset( new WLReaderMAT( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    WLIOStatus::ioStatus_t status;
    status = reader->init();
    if( status != WLIOStatus::SUCCESS )
    {
        errorLog() << reader->getIOStatusDescription( status );
        return false;
    }

    m_matrix.reset();
    status = reader->readMatrix( m_matrix );
    if( status != WLIOStatus::SUCCESS )
    {
        errorLog() << reader->getIOStatusDescription( status );
        return false;
    }
    reader->close();
    infoLog() << SUCCESS_READ << " Matrix: " << m_matrix->rows() << "x" << m_matrix->cols();
    return true;
}

bool WMMatReader::handleGenerateEMM()
{
    if( !m_matrix )
    {
        errorLog() << "No data available! First open a MAT-File.";
        m_trgGenerate->set( WPVBaseTypes::PV_TRIGGER_READY, true );
        return false;
    }

    WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
    WLEMMSubject::SPtr subject( new WLEMMSubject() );
    emm->setSubject( subject );

    WLEMDEEG::SPtr eeg( new WLEMDEEG() );
    eeg->setData( m_matrix );
    eeg->setSampFreq( m_propSamplFreq->get() );
    if( m_sensorPos.get() != NULL )
    {
        if( m_sensorPos->size() == eeg->getNrChans() )
        {
            infoLog() << "Set sensor positions for EEG.";
            eeg->setChannelPositions3d( m_sensorPos );
        }
        else
        {
            warnLog() << "EEG channels does not match positions size!";
        }
    }
    if( m_leadfield.get() != NULL )
    {
        if( m_leadfield->rows() == eeg->getNrChans() )
        {
            infoLog() << "Set leadfield for EEG.";
            subject->setLeadfield( WEModalityType::EEG, m_leadfield );
        }
        else
        {
            warnLog() << "EEG channels does not match Leadfield rows!";
        }
    }
    if( m_surface.get() != NULL )
    {
        infoLog() << "Set source space for EEG.";
        subject->setSurface( m_surface );
    }
    emm->addModality( eeg );

    infoLog() << SUCCESS_EMM << " EEG: " << eeg->getNrChans() << "x" << eeg->getSamplesPerChan();

    m_trgGenerate->set( WPVBaseTypes::PV_TRIGGER_READY, true );
    return processCompute( emm );
}

bool WMMatReader::handleLfFileChanged()
{
    const std::string fName = m_propLfFile->get().string();
    WLReaderLeadfield::SPtr reader;
    try
    {
        reader.reset( new WLReaderLeadfield( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    WLIOStatus::ioStatus_t state = reader->read( m_leadfield );
    if( state == WLIOStatus::SUCCESS )
    {
        infoLog() << SUCCESS_READ << " (Leadfield)";
        return true;
    }
    else
    {
        errorLog() << reader->getIOStatusDescription( state );
        return false;
    }
}

bool WMMatReader::handleSurfaceFileChanged()
{
    const std::string fName = m_propSrcSpaceFile->get().string();
    WLReaderSourceSpace::SPtr reader;
    try
    {
        reader.reset( new WLReaderSourceSpace( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        return false;
    }

    WLIOStatus::ioStatus_t state = reader->read( m_surface );
    if( state == WLIOStatus::SUCCESS )
    {
        infoLog() << SUCCESS_READ << " (Source Space)";
        return true;
    }
    else
    {
        errorLog() << reader->getIOStatusDescription( state );
        return false;
    }
}

bool WMMatReader::processCompute( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    cmd->setEmm( emm );
    m_output->updateData( cmd );
    return true;
}
