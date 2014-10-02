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

#include <cstdlib> // srand()
#include <ctime> // time()
#include <exception>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyTypes.h>
#include <core/common/WRealtimeTimer.h>

#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLEMMSurface.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLDataTypes.h"

#include "core/io/WLReaderELC.h"
#include "core/io/WLReaderFIFF.h"
#include "core/io/WLReaderDIP.h"
#include "core/io/WLReaderVOL.h"
#include "core/io/WLReaderExperiment.h"
#include "core/util/WLGeometry.h"

#include "WMEmMeasurement.h"
#include "WMEmMeasurement.xpm"

using std::string;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEmMeasurement )

static const std::string NO_DATA_LOADED = "No data loaded.";
static const std::string LOADING_DATA = "Loading data ...";
static const std::string DATA_LOADED = "Data successfully loaded.";
static const std::string DATA_ERROR = "Could not load data.";

static const std::string NO_FILE_LOADED = "No file loaded.";
static const std::string LOADING_FILE = "Loading file ...";
static const std::string FILE_LOADED = "File successfully loaded.";
static const std::string FILE_ERROR = "Could not load file.";

WMEmMeasurement::WMEmMeasurement()
{
    m_fiffEmm = WLEMMeasurement::SPtr( new WLEMMeasurement() );
    m_isDipLoaded = false;
    m_isElcLoaded = false;
    m_isExpLoaded = false;
    m_isFiffLoaded = false;
    m_isVolLoaded = false;
}

WMEmMeasurement::~WMEmMeasurement()
{
}

WModule::SPtr WMEmMeasurement::factory() const
{
    return WModule::SPtr( new WMEmMeasurement() );
}

const char** WMEmMeasurement::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEmMeasurement::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " EM-Measurement";
}

const std::string WMEmMeasurement::getDescription() const
{
    return "Entry point for LaBP data processing.";
}

void WMEmMeasurement::connectors()
{
    WLModuleDrawable::connectors();

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMEmMeasurement::properties()
{
    WLModuleDrawable::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    // Experiment loader - Fiff properties //
//    m_propGrpFiffStreaming = m_properties->addPropertyGroup( "FIFF streaming properties",
//                    "Contains properties for streaming data out of a fiff file", false );
    m_propGrpExperiment = m_properties->addPropertyGroup( "LaBP Experiment Loader", "LaBP Experiment Loader", false );

    m_fiffFileStatus = m_propGrpExperiment->addProperty( "FIFF file status:", "FIFF file status.", NO_FILE_LOADED );
    m_fiffFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_fiffFile = m_propGrpExperiment->addProperty( "FIFF file:", "Read a FIFF file for the data stream.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_fiffFile->changed( true );

    m_streamFiffTrigger = m_propGrpExperiment->addProperty( "FIFF streaming", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_streamFiffTrigger->setHidden( true );

    m_fiffStreamBlockSize = m_propGrpExperiment->addProperty( "Block size", "Block size for data generator in milliseconds",
                    1000 );
    m_fiffStreamBlockSize->setMin( 100 );
    m_fiffStreamBlockSize->setMax( 120000 );

    // Experiment loader - Metadata //
    m_expLoadStatus = m_propGrpExperiment->addProperty( "Metadata status:", "LaBP Experiment data status.", NO_DATA_LOADED );
    m_expLoadStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_expLoadTrigger = m_propGrpExperiment->addProperty( "Load data", "Load data", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_expLoadTrigger->setHidden( true );

    const std::string subject( "none" );
    m_expSubject = m_propGrpExperiment->addProperty( "Subject:", "Selected a subject.", subject );
    m_expSubject->setHidden( true );
    m_expSubject->setPurpose( PV_PURPOSE_INFORMATION );

    const std::string trial( "none" );
    m_expTrial = m_propGrpExperiment->addProperty( "Trial:", "Selected a trial.", trial );
    m_expTrial->setHidden( true );
    m_expTrial->setPurpose( PV_PURPOSE_INFORMATION );

    m_expBemFiles = boost::shared_ptr< WItemSelection >( new WItemSelection() );

    m_expSurfaces = boost::shared_ptr< WItemSelection >( new WItemSelection() );

    // Generation properties //
    m_propGrpDataGeneration = m_properties->addPropertyGroup( "Data generation properties",
                    "Contains properties for random data generation", false );

    m_genDataTrigger = m_propGrpDataGeneration->addProperty( "Data generator", "Start", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );
    m_genDataTriggerEnd = m_propGrpDataGeneration->addProperty( "Data generator ", "Stop", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition, true );
    m_generationFreq = m_propGrpDataGeneration->addProperty( "Sampling frequency in Hz",
                    "Sampling frequency to generate Data with", 1000 );
    m_generationFreq->setMax( 10000 );
    m_generationFreq->setMin( 10 );
    m_generationBlockSize = m_propGrpDataGeneration->addProperty( "Block length in ms",
                    "Block size for the data generator in milliseconds", 1000 );
    m_generationBlockSize->setMin( 100 );
    m_generationBlockSize->setMax( 10000 );

    m_generationDuration = m_propGrpDataGeneration->addProperty( "Length of data stream in s",
                    "Duration to generate Data in seconds", 60 );
    m_generationDuration->setMax( 600 ); // 10 minutes
    m_generationDuration->setMin( 10 ); // 10 seconds
    m_generationNrChans = m_propGrpDataGeneration->addProperty( "Number of Channels", "Number of channels to generate", 100 );
    m_generationNrChans->setMin( 10 );
    m_generationNrChans->setMax( 50000 );

    // ELC Propeties //
    m_propGrpExtra = m_properties->addPropertyGroup( "Additional information", "Options to load and set additional information.",
                    false );
    m_elcFileStatus = m_propGrpExtra->addProperty( "ELC file status:", "ELC file status.", NO_FILE_LOADED );
    m_elcFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_elcFile = m_propGrpExtra->addProperty( "ELC file:", "Read a ELC file.", WPathHelper::getHomePath(), m_propCondition );
    m_elcFile->changed( true );
    m_elcChanLabelCount = m_propGrpExtra->addProperty( "Channel labels:", "Found channel labels in file.", 0 );
    m_elcChanLabelCount->setPurpose( PV_PURPOSE_INFORMATION );
    m_elcChanPositionCount = m_propGrpExtra->addProperty( "Channel positions:", "Found channel positions in file.", 0 );
    m_elcChanPositionCount->setPurpose( PV_PURPOSE_INFORMATION );
    m_elcFacesCount = m_propGrpExtra->addProperty( "Faces:", "Found faces in file.", 0 );
    m_elcFacesCount->setPurpose( PV_PURPOSE_INFORMATION );

    // DIP Properties //
    m_dipFileStatus = m_propGrpExtra->addProperty( "DIP file status:", "DIP file status.", NO_FILE_LOADED );
    m_dipFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_dipFile = m_propGrpExtra->addProperty( "DIP file:", "Reads a DIP file.", WPathHelper::getHomePath(), m_propCondition );
    m_dipFile->changed( true );
    m_dipPositionCount = m_propGrpExtra->addProperty( "DIP Positions:", "Found positions in file.", 0 );
    m_dipPositionCount->setPurpose( PV_PURPOSE_INFORMATION );
    m_dipFacesCount = m_propGrpExtra->addProperty( "DIP Faces:", "Found faces in file.", 0 );
    m_dipFacesCount->setPurpose( PV_PURPOSE_INFORMATION );

    // VOL Properties //
    m_volFileStatus = m_propGrpExtra->addProperty( "VOL file status:", "VOL file status.", NO_FILE_LOADED );
    m_volFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_volFile = m_propGrpExtra->addProperty( "VOL file:", "Read a VOL file.", WPathHelper::getHomePath(), m_propCondition );
    m_volFile->changed( true );
    m_volBoundaryCount = m_propGrpExtra->addProperty( "BEM Boundaries:", "BEM Boundaries found in file.", 0 );
    m_volBoundaryCount->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMEmMeasurement::moduleInit()
{
    infoLog() << "Initializing module ...";
    waitRestored();
    srand( time( NULL ) );
    m_isFiffLoaded = false;
    m_isElcLoaded = false;
    m_isDipLoaded = false;
    m_isExpLoaded = false;

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC );
    infoLog() << "Initializing module finished!";
}

void WMEmMeasurement::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );
    m_moduleState.add( m_genDataTrigger->getCondition() );
    m_moduleState.add( m_genDataTriggerEnd->getCondition() );

    ready();

    moduleInit();

    while( !m_shutdownFlag() )
    {
        debugLog() << "Waiting for Events";
        m_moduleState.wait(); // wait for events like inputdata or properties changed

        if( m_shutdownFlag() )
        {
            break;
        }

        if( ( m_genDataTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            generateData();
        }

        if( m_fiffFile->changed( true ) )
        {
            m_isFiffLoaded = readFiff( m_fiffFile->get().string() );
            extractExpLoader( m_fiffFile->get().string() );
        }

        if( m_elcFile->changed( true ) )
        {
            m_isElcLoaded = readElc( m_elcFile->get().string() );
        }

        if( m_dipFile->changed( true ) )
        {
            m_isDipLoaded = readDip( m_dipFile->get().string() );
        }

        if( m_volFile->changed( true ) )
        {
            m_isVolLoaded = readVol( m_volFile->get().string() );
        }

        if( ( m_streamFiffTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            streamData();
        }

        if( ( m_expLoadTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            handleExperimentLoadChanged();
        }
    }

    viewCleanup();
}

void WMEmMeasurement::streamData()
{
    viewReset();
    if( m_isFiffLoaded )
    {
        infoLog() << "Streaming started ...";
        m_fiffFile->setHidden( true );

        WRealtimeTimer totalTimer;
        WRealtimeTimer waitTimer;

        const double SEC_PER_BLOCK = ( double )m_fiffStreamBlockSize->get() / 1000; // blockSize in seconds
        WLSampleNrT blockSize = 0; // blockSize depending on SEC_PER_BLOCK and sampling frequency
        WLSampleNrT blockOffset = 0; // start index for current block
        size_t blockCount = 0;
        int smplFrq;
        WLEMData::DataT fiffData;
        bool hasData;
        std::vector< WLEMData::SPtr > emds = m_fiffEmm->getModalityList();
        boost::shared_ptr< std::vector< std::vector< int > > > events = m_fiffEmm->getEventChannels();

        totalTimer.reset();

        // start streaming
        do
        {
            hasData = false;

            if( m_shutdownFlag() )
            {
                break;
            }

            waitTimer.reset();

            WLEMMeasurement::SPtr emmPacket( new WLEMMeasurement( *m_fiffEmm ) );

            // clone each modality
            for( std::vector< WLEMData::SPtr >::const_iterator emd = emds.begin(); emd != emds.end(); ++emd )
            {
                if( m_shutdownFlag() )
                {
                    break;
                }

                smplFrq = ( *emd )->getSampFreq();
                blockSize = smplFrq * SEC_PER_BLOCK;
                blockOffset = blockCount * blockSize;
                WLEMData::SPtr emdPacket = ( *emd )->clone();
                WLEMData::DataSPtr data( new WLEMData::DataT( ( *emd )->getNrChans(), blockSize ) );
                fiffData = ( *emd )->getData();

                // copy each channel
                for( WLChanIdxT chan = 0; chan < ( *emd )->getNrChans(); ++chan )
                {
                    for( WLSampleIdxT sample = 0; sample < blockSize && ( blockOffset + sample ) < fiffData.cols(); ++sample )
                    {
                        ( *data )( chan, sample ) = fiffData( chan, blockOffset + sample );
                    }
                }

                emdPacket->setData( data );
                emmPacket->addModality( emdPacket );

                if( ( *emd )->getNrChans() > 0 )
                {
                    debugLog() << "emdPacket type: " << emdPacket->getModalityType() << " size: " << emdPacket->getData().cols();

                    // set termination condition
                    hasData = hasData || blockOffset + blockSize < fiffData.cols();
                }
            }

            // copy event channels
            blockOffset = blockCount * blockSize; // Using blockSize/samplFreq of last modality
            for( WLEMMeasurement::EDataT::const_iterator eChannel = events->begin(); eChannel != events->end(); ++eChannel )
            {
                WLEMMeasurement::EChannelT data;
                data.reserve( blockSize );
                for( WLSampleIdxT event = 0; event < blockSize && ( blockOffset + event ) < ( *eChannel ).size(); ++event )
                {
                    data.push_back( ( *eChannel )[blockOffset + event] );
                }
                emmPacket->addEventChannel( data );

                debugLog() << "emmPacket event size: " << data.size();
            }

            setAdditionalInformation( emmPacket );

            debugLog() << "emmPacket modalities: " << emmPacket->getModalityCount();
            debugLog() << "emmPacket events: " << emmPacket->getEventChannelCount();

            processCompute( emmPacket );

            ++blockCount;
            infoLog() << "Streamed emmPacket #" << blockCount;

            // waiting to simulate streaming
            const double tuSleep = SEC_PER_BLOCK * 1000000 - waitTimer.elapsed() * 1000000;
            if( tuSleep > 0 )
            {
                boost::this_thread::sleep( boost::posix_time::microseconds( tuSleep ) );
                debugLog() << "Slept for " << tuSleep << " microseconds.";
            }
            else
            {
                warnLog() << "Streaming took " << abs( tuSleep ) << " microseconds to long!";
            }
        } while( hasData );

        const double total = totalTimer.elapsed() * 1000;

        infoLog() << "Streaming finished!";
        debugLog() << "Generation time: " << total << " ms";
    }
    else
    {
        warnLog() << "No FIFF file loaded!";
    }
    m_fiffFile->setHidden( false );
    m_streamFiffTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, false );
}

void WMEmMeasurement::generateData()
{
    viewReset();
    infoLog() << "Generation started ...";
    m_genDataTrigger->setHidden( true );
    m_genDataTriggerEnd->setHidden( false );

    WRealtimeTimer totalTimer;
    WRealtimeTimer waitTimer;

    totalTimer.reset();

    for( int k = 0; k < ( ( double )m_generationDuration->get() * 1000 ) / ( double )m_generationBlockSize->get(); k++ )
    {
        if( m_shutdownFlag() )
        {
            break;
        }

        waitTimer.reset();

        if( ( m_genDataTriggerEnd->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            m_genDataTriggerEnd->set( WPVBaseTypes::PV_TRIGGER_READY, true );
            infoLog() << "Generation stopped ...";
            break;
        }

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WLEMDEEG::SPtr eeg( new WLEMDEEG() );

        eeg->setSampFreq( m_generationFreq->get() );

        const size_t channels = m_generationNrChans->get();
        const size_t samples = m_generationFreq->get() * ( ( double )m_generationBlockSize->get() / 1000.0 );
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
        setAdditionalInformation( emm );

        processCompute( emm );

        debugLog() << "inserted block " << k + 1 << "/"
                        << ( ( double )m_generationDuration->get() * 1000 ) / ( double )m_generationBlockSize->get() << " with "
                        << samples << " Samples";

        const double tuSleep = m_generationBlockSize->get() * 1000 - ( waitTimer.elapsed() * 1000000 );
        if( tuSleep > 0 )
        {
            boost::this_thread::sleep( boost::posix_time::microseconds( tuSleep ) );
            debugLog() << "Slept for " << tuSleep << " microseconds.";
        }
        else
        {
            warnLog() << "Generation took " << abs( tuSleep ) << " microseconds to long!";
        }
    }

    const double total = totalTimer.elapsed() * 1000;

    infoLog() << "Generation finished!";
    debugLog() << "Generation time: " << total << " ms";

    m_genDataTriggerEnd->setHidden( true );
    m_genDataTrigger->setHidden( false );

    m_genDataTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

bool WMEmMeasurement::readFiff( std::string fname )
{
    infoLog() << "Reading FIFF file: " << fname;
    m_fiffFileStatus->set( LOADING_FILE, true );
    if( boost::filesystem::exists( fname ) && boost::filesystem::is_regular_file( fname ) )
    {
        WLReaderFIFF fiffReader( fname );
        m_fiffEmm.reset( new WLEMMeasurement() );
        if( fiffReader.Read( m_fiffEmm ) == WLReaderFIFF::ReturnCode::SUCCESS )
        {
            if( m_fiffEmm->hasModality( WLEModality::EEG ) )
            {
                WLEMDEEG::SPtr eeg = m_fiffEmm->getModality< WLEMDEEG >( WLEModality::EEG );
                if( eeg->getFaces()->empty() )
                {
                    warnLog() << "No faces found! Faces will be generated.";
                    WLGeometry::computeTriangulation( eeg->getFaces().get(), *eeg->getChannelPositions3d(), -5 );
                }
            }
            infoLog() << *m_fiffEmm;
            infoLog() << "Reading FIFF file finished!";
            m_fiffFileStatus->set( FILE_LOADED, true );
            m_streamFiffTrigger->setHidden( false );
            return true;
        }
        else
        {
            m_streamFiffTrigger->setHidden( true );
            warnLog() << "Could not read file! Maybe not in FIFF format.";
            m_fiffFileStatus->set( FILE_ERROR, true );
            return false;
        }
    }
    else
    {
        m_streamFiffTrigger->setHidden( true );
        warnLog() << "File does not exist!";
        m_fiffFileStatus->set( FILE_ERROR, true );
        return false;
    }
    return false;
}

bool WMEmMeasurement::readElc( std::string fname )
{
    m_elcFileStatus->set( LOADING_FILE, true );
    m_elcLabels = WLArrayList< std::string >::instance();
    m_elcPositions3d.reset( new std::vector< WPosition >() );
    m_elcFaces.reset( new std::vector< WVector3i >() );

    WLReaderELC* elcReader;
    try
    {
        elcReader = new WLReaderELC( fname );
    }
    catch( const std::exception& e )
    {
        errorLog() << e.what();
        m_elcFileStatus->set( FILE_ERROR, true );
        return false;
    }

    if( elcReader->read( m_elcPositions3d, m_elcLabels, m_elcFaces ) == WLReaderELC::ReturnCode::SUCCESS )
    {
        m_elcChanLabelCount->set( m_elcLabels->size(), true );
        m_elcChanPositionCount->set( m_elcPositions3d->size(), true );
        m_elcFacesCount->set( m_elcFaces->size(), true );

#ifdef DEBUG
        debugLog() << "First data read from elc file ... ";
        std::stringstream stream;
        stream << "ELC positions: ";
        for( size_t i = 0; i < 5 && i < m_elcPositions3d->size(); ++i )
        {
            stream << m_elcPositions3d->at( i ) << " ";
        }
        debugLog() << stream.str();

        stream.clear();
        stream.str( "" );
        stream << "ELC labels: ";
        for( size_t i = 0; i < 5 && i < m_elcLabels->size(); ++i )
        {
            stream << m_elcLabels->at( i ) << " ";
        }
        debugLog() << stream.str();

        stream.clear();
        stream.str( "" );
        stream << "ELC faces: ";
        for( size_t i = 0; i < 5 && i < m_elcFaces->size(); ++i )
        {
            stream << m_elcFaces->at( i ) << " ";
        }
        debugLog() << stream.str();
#endif // DEBUG
        m_elcFileStatus->set( FILE_LOADED, true );
        delete elcReader;
        return true;
    }
    else
    {
        warnLog() << "Could not read file! Maybe not in elc format.";
        m_elcFileStatus->set( FILE_ERROR, true );

        delete elcReader;
        return false;
    }
}

bool WMEmMeasurement::readDip( std::string fname )
{
    m_dipFileStatus->set( LOADING_FILE, true );
    m_dipSurface.reset();

    WLReaderDIP* reader;
    try
    {
        reader = new WLReaderDIP( fname );
    }
    catch( const std::exception& e )
    {
        errorLog() << e.what();
        m_dipFileStatus->set( FILE_ERROR, true );
        return false;
    }

    m_dipSurface.reset( new WLEMMSurface() );
    if( reader->read( m_dipSurface ) == WLReaderDIP::ReturnCode::SUCCESS )
    {
        m_dipPositionCount->set( m_dipSurface->getVertex()->size(), true );
        m_dipFacesCount->set( m_dipSurface->getFaces()->size(), true );

#ifdef DEBUG
        debugLog() << "First data read from dip file ... ";
        std::stringstream stream;
        stream << "DIP positions: ";
        for( size_t i = 0; i < 5 && i < m_dipSurface->getVertex()->size(); ++i )
        {
            stream << m_dipSurface->getVertex()->at( i ) << " ";
        }
        debugLog() << stream.str();

        stream.clear();
        stream.str( "" );
        stream << "DIP faces: ";
        for( size_t i = 0; i < 5 && i < m_dipSurface->getFaces()->size(); ++i )
        {
            stream << m_dipSurface->getFaces()->at( i ) << " ";
        }
        debugLog() << stream.str();
#endif // DEBUG
        m_dipFileStatus->set( FILE_LOADED, true );
        delete reader;
        return true;
    }
    else
    {
        warnLog() << "Could not read file! Maybe not in dip format.";
        m_dipFileStatus->set( FILE_ERROR, true );

        delete reader;
        return false;
    }
}

bool WMEmMeasurement::readVol( std::string fname )
{
    m_volFileStatus->set( LOADING_FILE, true );
    m_volBoundaries.reset();

    WLReaderVOL* reader;
    try
    {
        reader = new WLReaderVOL( fname );
    }
    catch( const std::exception& e )
    {
        errorLog() << e.what();
        m_volFileStatus->set( FILE_ERROR, true );
        return false;
    }

    m_volBoundaries = WLList< WLEMMBemBoundary::SPtr >::instance();
    if( reader->read( m_volBoundaries.get() ) == WLReaderVOL::ReturnCode::SUCCESS )
    {
        m_volBoundaryCount->set( m_volBoundaries->size(), true );
        m_volFileStatus->set( FILE_LOADED, true );
        delete reader;
        return true;
    }
    else
    {
        warnLog() << "Could not read file! Maybe not in vol format.";
        m_volFileStatus->set( FILE_ERROR, true );

        delete reader;
        return false;
    }
}

void WMEmMeasurement::setAdditionalInformation( WLEMMeasurement::SPtr emm )
{
    if( m_isElcLoaded )
    {
        std::vector< WLEMData::SPtr > modalities = emm->getModalityList();
        for( std::vector< WLEMData::SPtr >::iterator it = modalities.begin(); it != modalities.end(); ++it )
        {
            ( *it )->setChanNames( m_elcLabels ); // TODO(pieloth) m_elcLabels are specific for each modality
            if( ( *it )->getModalityType() == WLEModality::EEG )
            {
                WLEMDEEG::SPtr eeg = ( *it )->getAs< WLEMDEEG >();
                eeg->setFaces( m_elcFaces );
                eeg->setChannelPositions3d( m_elcPositions3d );
            }
        }
    }
    if( m_isExpLoaded )
    {
        emm->setSubject( m_subject );
    }
    else
    {
        if( m_isDipLoaded )
        {
            emm->getSubject()->setSurface( m_dipSurface );
        }
        if( m_isVolLoaded )
        {
            emm->getSubject()->setBemBoundaries( m_volBoundaries );
        }
    }
}

void WMEmMeasurement::handleExperimentLoadChanged()
{
    debugLog() << "handleExperimentLoadChanged() called!";

    bool rc = false;
    m_expLoadStatus->set( LOADING_DATA, true );

    m_subject.reset( new WLEMMSubject() );

    const string bemFile = m_expBemFilesSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    rc |= m_expReader->readBem( bemFile, m_subject );

    const string surfaceType = m_expSurfacesSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    rc |= m_expReader->readSourceSpace( surfaceType, m_subject );

    const string trial = m_expTrial->get();
    rc |= m_expReader->readLeadFields( surfaceType, bemFile, trial, m_subject );

    if( rc )
    {
        m_expLoadStatus->set( DATA_LOADED, true );
        m_isExpLoaded = true;

        WLEMMeasurement::SPtr emm( new WLEMMeasurement( m_subject ) );
        WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::INIT );
        labp->setEmm( emm );
        m_output->updateData( labp );
    }
    else
    {
        m_expLoadStatus->set( DATA_ERROR, true );
        m_isExpLoaded = false;
    }

    m_expLoadTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMEmMeasurement::extractExpLoader( std::string fName )
{
    boost::filesystem::path fiffFile( fName );
    boost::filesystem::path expRoot = WLReaderExperiment::getExperimentRootFromFiff( fiffFile );
    const std::string subject = WLReaderExperiment::getSubjectFromFiff( fiffFile );
    const std::string trial = WLReaderExperiment::getTrialFromFiff( fiffFile );

    m_expSubject->set( subject );
    m_expSubject->setHidden( false );
    m_expTrial->set( trial );
    m_expTrial->setHidden( false );

    if( !boost::filesystem::is_directory( expRoot ) )
    {
        expRoot = expRoot.parent_path();
    }
    m_expReader.reset( new WLReaderExperiment( expRoot.string(), subject ) );

    std::set< std::string > bems = m_expReader->findBems();
    m_expBemFiles->clear();
    m_propGrpExperiment->removeProperty( m_expBemFilesSelection );
    if( !bems.empty() )
    {
        for( std::set< std::string >::iterator it = bems.begin(); it != bems.end(); ++it )
        {
            m_expBemFiles->addItem(
                            WItemSelectionItemTyped< string >::SPtr( new WItemSelectionItemTyped< string >( *it, *it ) ) );
        }
        m_expBemFilesSelection = m_propGrpExperiment->addProperty( "BEM Layers:", "Select BEM Layers to use.",
                        m_expBemFiles->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_expBemFilesSelection );
        m_expLoadTrigger->setHidden( false );
    }

    m_expSurfaces->clear();
    m_propGrpExperiment->removeProperty( m_expSurfacesSelection );
    std::set< std::string > surfaces = m_expReader->findSurfaceKinds();
    if( !surfaces.empty() )
    {
        for( std::set< std::string >::iterator it = surfaces.begin(); it != surfaces.end(); ++it )
        {
            m_expSurfaces->addItem(
                            WItemSelectionItemTyped< string >::SPtr( new WItemSelectionItemTyped< string >( *it, *it ) ) );
        }
        m_expSurfacesSelection = m_propGrpExperiment->addProperty( "Surfaces:", "Select surface to use.",
                        m_expSurfaces->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_expSurfacesSelection );
        m_expLoadTrigger->setHidden( false );
    }
}

bool WMEmMeasurement::processCompute( WLEMMeasurement::SPtr emm )
{
    // Set a new profiler for the new EMM
    emm->setProfiler( WLLifetimeProfiler::instance( WLEMMeasurement::CLASS, "lifetime" ) );

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    labp->setEmm( emm );
    viewUpdate( emm );
    m_output->updateData( labp );
    return true;
}

bool WMEmMeasurement::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMEmMeasurement::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMEmMeasurement::processMisc( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}
