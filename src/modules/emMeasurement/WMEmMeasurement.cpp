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

// Output connector and data
// TODO use OW class
#include "core/kernel/WLModuleOutputDataCollectionable.h"
#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEEG.h"
#include "core/dataHandler/WDataSetEMMMEG.h"
#include "core/dataHandler/WDataSetEMMSubject.h"
#include "core/dataHandler/WDataSetEMMSurface.h"
#include "core/dataHandler/WDataSetEMMBemBoundary.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "core/io/WLReaderELC.h"
#include "core/io/WLReaderFIFF.h"
#include "core/io/WLReaderDIP.h"
#include "core/io/WLReaderVOL.h"
#include "core/io/WLReaderExperiment.h"
#include "core/util/WLGeometry.h"

#include "WMEmMeasurement.h"
#include "WMEmMeasurement.xpm"

using namespace boost;
using namespace std;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEmMeasurement )

WMEmMeasurement::WMEmMeasurement()
{
    m_fiffEmm = boost::shared_ptr< LaBP::WDataSetEMM >( new LaBP::WDataSetEMM() );
}

WMEmMeasurement::~WMEmMeasurement()
{

}

boost::shared_ptr< WModule > WMEmMeasurement::factory() const
{
    return boost::shared_ptr< WModule >( new WMEmMeasurement() );
}

const char** WMEmMeasurement::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEmMeasurement::getName() const
{
    return "EM-Measurement";
}

const std::string WMEmMeasurement::getDescription() const
{
    return "Entry point for LaBP data processing.";
}

void WMEmMeasurement::connectors()
{
    // initialize connectors
    // TODO use OW class
    m_output = boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM > >(
                    new LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM >( shared_from_this(), "out",
                                    "A loaded dataset." ) );

    // add it to the list of connectors. Please note, that a connector NOT added via addConnector will not work as expected.
    addConnector( m_output );
}

void WMEmMeasurement::properties()
{
    LaBP::WLModuleDrawable::properties();

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

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

    // Registration Properties //
    m_propGrpRegistration = m_properties->addPropertyGroup( "Registration and Alignment",
                    "Alignment of different coordinate systems.", false );
    m_regAlignTrigger = m_propGrpRegistration->addProperty( "Start Alignment", "Start Alignment", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );

    m_regError = m_propGrpRegistration->addProperty( "Deviation:", "Deviation after alignment.", -1.0 );
    m_regError->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMEmMeasurement::initModule()
{
    infoLog() << "Initializing module ...";
    waitRestored();
    srand( time( NULL ) );
    m_isFiffLoaded = false;
    m_isElcLoaded = false;
    m_isDipLoaded = false;
    m_isExpLoaded = false;

    initView( LaBP::WLEMDDrawable2D::WEGraphType::DYNAMIC );
    infoLog() << "Initializing module finished!";
}

void WMEmMeasurement::moduleMain()
{
    m_moduleState.setResetable( true, true );
    m_moduleState.add( m_propCondition );
    m_moduleState.add( m_genDataTrigger->getCondition() );
    m_moduleState.add( m_genDataTriggerEnd->getCondition() );

    ready();

    initModule();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
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

        if( ( m_regAlignTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            align();
        }

        if( ( m_expLoadTrigger->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED ) )
        {
            handleExperimentLoadChanged();
        }
    }
}

void WMEmMeasurement::streamData()
{
    resetView();
    if( m_isFiffLoaded )
    {
        infoLog() << "Streaming started ...";
        m_fiffFile->setHidden( true );

        WRealtimeTimer totalTimer;
        WRealtimeTimer waitTimer;

        const double SEC_PER_BLOCK = ( double )m_fiffStreamBlockSize->get() / 1000; // blockSize in seconds
        size_t blockSize = 0; // blockSize depending on SEC_PER_BLOCK and sampling frequency
        size_t blockOffset = 0; // start index for current block
        size_t blockCount = 0;
        int smplFrq;
        std::vector< std::vector< double > > fiffData;
        bool hasData;
        std::vector< LaBP::WDataSetEMMEMD::SPtr > emds = m_fiffEmm->getModalityList();
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

            boost::shared_ptr< LaBP::WDataSetEMM > emmPacket = boost::shared_ptr< LaBP::WDataSetEMM >(
                            new LaBP::WDataSetEMM( *m_fiffEmm ) );

            // clone each modality
            for( std::vector< LaBP::WDataSetEMMEMD::SPtr >::const_iterator emd = emds.begin(); emd != emds.end(); ++emd )
            {
                if( m_shutdownFlag() )
                {
                    break;
                }

                boost::shared_ptr< LaBP::WDataSetEMMEMD > emdPacket = ( *emd )->clone();
                boost::shared_ptr< std::vector< std::vector< double > > > data( new std::vector< std::vector< double > >() );
                data->reserve( ( *emd )->getNrChans() );

                smplFrq = ( *emd )->getSampFreq();
                blockSize = smplFrq * SEC_PER_BLOCK;
                blockOffset = blockCount * blockSize;
                fiffData = ( *emd )->getData();

                // copy each channel
                for( size_t chan = 0; chan < ( *emd )->getNrChans(); ++chan )
                {
                    std::vector< double > channel;
                    channel.reserve( blockSize );

                    for( size_t sample = 0; sample < blockSize && ( blockOffset + sample ) < fiffData.at( chan ).size();
                                    ++sample )
                    {
                        channel.push_back( fiffData[chan][blockOffset + sample] );
                    }

                    data->push_back( channel );
                }

                emdPacket->setData( data );
                emmPacket->addModality( emdPacket );

                if( ( *emd )->getNrChans() > 0 )
                {
                    debugLog() << "emdPacket type: " << emdPacket->getModalityType() << " size: "
                                    << emdPacket->getData().front().size();

                    // set termination condition
                    hasData = hasData || blockOffset + blockSize < fiffData[0].size();
                }
            }

            // copy event channels
            blockOffset = blockCount * blockSize; // Using blockSize/samplFreq of last modality
            for( std::vector< std::vector< int > >::const_iterator eChannel = events->begin(); eChannel != events->end();
                            ++eChannel )
            {
                std::vector< int > data;
                data.reserve( blockSize );
                for( size_t event = 0; event < blockSize && ( blockOffset + event ) < ( *eChannel ).size(); ++event )
                {
                    data.push_back( ( *eChannel )[blockOffset + event] );
                }
                emmPacket->addEventChannel( data );

                debugLog() << "emmPacket event size: " << data.size();
            }

            setAdditionalInformation( emmPacket );

            debugLog() << "emmPacket modalities: " << emmPacket->getModalityCount();
            debugLog() << "emmPacket events: " << emmPacket->getEventChannelCount();

            emmPacket->getTimeProfiler()->start();
            updateView( emmPacket ); // update view
            m_output->updateData( emmPacket ); // update connected modules

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
    resetView();
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

        boost::shared_ptr< LaBP::WDataSetEMM > emm = boost::shared_ptr< LaBP::WDataSetEMM >( new LaBP::WDataSetEMM() );
        boost::shared_ptr< LaBP::WDataSetEMMEEG > eeg = boost::shared_ptr< LaBP::WDataSetEMMEEG >( new LaBP::WDataSetEMMEEG() );

        eeg->setSampFreq( m_generationFreq->get() );

        for( int i = 0; i < m_generationNrChans->get(); i++ )
        {
            std::vector< double > channel;
            for( int j = 0; j < m_generationFreq->get() * ( ( double )m_generationBlockSize->get() / 1000.0 ); j++ )
            {
                channel.push_back( 30.0 * ( double )rand() / RAND_MAX - 15.0 );
            }
            double a = ( double )rand() / RAND_MAX - 0.5;
            double b = ( double )rand() / RAND_MAX - 0.5;
            double c = ( double )rand() / RAND_MAX - 0.5;
            double m = sqrt( a * a + b * b + c * c );
            double r = 100;
            a *= r / m;
            b *= r / m;
            c *= r / m;
            //WPosition point = new ;
            eeg->getChannelPositions3d()->push_back( WPosition( a, b, abs( c ) ) );
            eeg->getData().push_back( channel );
        }

        emm->addModality( eeg );
        setAdditionalInformation( emm );

        emm->getTimeProfiler()->start();
        updateView( emm );

        m_output->updateData( emm );
        debugLog() << "m_output->updateData() called!";

        debugLog() << "inserted block " << k + 1 << "/"
                        << ( ( double )m_generationDuration->get() * 1000 ) / ( double )m_generationBlockSize->get() << " with "
                        << m_generationFreq->get() * ( double )m_generationBlockSize->get() / 1000.0 << " Samples";

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
        LaBP::WLReaderFIFF fiffReader( fname );
        m_fiffEmm.reset( new LaBP::WDataSetEMM() );
        if( fiffReader.Read( m_fiffEmm ) == LaBP::WLReaderFIFF::ReturnCode::SUCCESS )
        {
            if( m_fiffEmm->hasModality( LaBP::WEModalityType::EEG ) )
            {
                LaBP::WDataSetEMMEEG::SPtr eeg = m_fiffEmm->getModality< LaBP::WDataSetEMMEEG >( LaBP::WEModalityType::EEG );
                if( eeg->getFaces().empty() )
                {
                    warnLog() << "No faces found! Faces will be generated.";
                    WLGeometry::computeTriangulation( eeg->getFaces(), *eeg->getChannelPositions3d() );
                }
            }
            infoLog() << "Modalities:\t" << m_fiffEmm->getModalityCount();
            infoLog() << "Event channels:\t" << m_fiffEmm->getEventChannelCount();
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
}

bool WMEmMeasurement::readElc( std::string fname )
{
    m_elcFileStatus->set( LOADING_FILE, true );
    m_elcLabels.reset( new std::vector< std::string >() );
    m_elcPositions3d.reset( new std::vector< WPosition >() );
    m_elcFaces.reset( new std::vector< WVector3i >() );

    LaBP::WLReaderELC* elcReader;
    try
    {
        elcReader = new LaBP::WLReaderELC( fname );
    }
    catch( std::exception& e )
    {
        errorLog() << e.what();
        m_elcFileStatus->set( FILE_ERROR, true );
        return false;
    }

    if( elcReader->read( m_elcPositions3d, m_elcLabels, m_elcFaces ) == LaBP::WLReaderELC::ReturnCode::SUCCESS )
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

    LaBP::WLReaderDIP* reader;
    try
    {
        reader = new LaBP::WLReaderDIP( fname );
    }
    catch( std::exception& e )
    {
        errorLog() << e.what();
        m_dipFileStatus->set( FILE_ERROR, true );
        return false;
    }

    m_dipSurface.reset( new LaBP::WDataSetEMMSurface() );
    if( reader->read( m_dipSurface ) == LaBP::WLReaderDIP::ReturnCode::SUCCESS )
    {
        m_dipPositionCount->set( m_dipSurface->getVertex()->size(), true );
        m_dipFacesCount->set( m_dipSurface->getFaces().size(), true );

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
        for( size_t i = 0; i < 5 && i < m_dipSurface->getFaces().size(); ++i )
        {

            stream << m_dipSurface->getFaces().at( i ) << " ";
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

    LaBP::WLReaderVOL* reader;
    try
    {
        reader = new LaBP::WLReaderVOL( fname );
    }
    catch( std::exception& e )
    {
        errorLog() << e.what();
        m_volFileStatus->set( FILE_ERROR, true );
        return false;
    }

    m_volBoundaries.reset( new std::vector< boost::shared_ptr< LaBP::WDataSetEMMBemBoundary > >() );
    if( reader->read( m_volBoundaries ) == LaBP::WLReaderVOL::ReturnCode::SUCCESS )
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

void WMEmMeasurement::setAdditionalInformation( LaBP::WDataSetEMM::SPtr emm )
{
    if( m_isElcLoaded )
    {
        std::vector< LaBP::WDataSetEMMEMD::SPtr > modalities = emm->getModalityList();
        for( std::vector< LaBP::WDataSetEMMEMD::SPtr >::iterator it = modalities.begin(); it != modalities.end();
                        ++it )
        {
            ( *it )->setChanNames( m_elcLabels ); // TODO m_elcLabels are specific for each modality
            if( ( *it )->getModalityType() == LaBP::WEModalityType::EEG )
            {
                LaBP::WDataSetEMMEEG::SPtr eeg = ( *it )->getAs< LaBP::WDataSetEMMEEG >();
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

void WMEmMeasurement::align()
{
    infoLog() << "Start alignment for FIFF file and EEG only!";
    if( m_isFiffLoaded )
    {
        if( m_fiffEmm->hasModality( LaBP::WEModalityType::EEG ) )
        {
            setAdditionalInformation( m_fiffEmm );

            LaBP::WDataSetEMMEEG::SPtr eeg = m_fiffEmm->getModality< LaBP::WDataSetEMMEEG >( LaBP::WEModalityType::EEG );
            boost::shared_ptr< std::vector< WPosition > > from = eeg->getChannelPositions3d();

            std::vector< LaBP::WDataSetEMMBemBoundary::SPtr > bems = m_fiffEmm->getSubject()->getBemBoundaries();
            LaBP::WDataSetEMMBemBoundary::SPtr bemSkin;
            for( std::vector< LaBP::WDataSetEMMBemBoundary::SPtr >::iterator it = bems.begin(); it != bems.end(); ++it )
            {
                if( ( *it )->getBemType() == LaBP::WEBemType::SKIN || ( *it )->getBemType() == LaBP::WEBemType::OUTER_SKIN )
                {
                    bemSkin = ( *it );
                    break;
                }
            }
            if( bemSkin )
            {
                std::vector< WPosition > to = bemSkin->getVertex();
                // TODO check unit!
                double error = m_regNaive.compute( *from, to );
                infoLog() << WRegistrationNaive::CLASS << " error: " << error;

                error = m_regICP.compute( *from, to, m_regNaive.getTransformationMatrix() );
                infoLog() << WRegistrationICP::CLASS << " error: " << error;

                m_regError->set( error, true );
                m_regTransformation = m_regICP.getTransformationMatrix();
                infoLog() << "Transformation: " << m_regTransformation;
            }
            else
            {
                errorLog() << "No BEM skin layer found. Alignment is canceled!";
                m_regError->set( -1.0, true );
            }

        }
        else
        {
            errorLog() << "No EEG modality found. Alignment is canceled!";
            m_regError->set( -1.0, true );
        }
    }
    else
    {
        errorLog() << "No FIFF file loaded. Alignment is canceled!";
        m_regError->set( -1.0, true );
    }

    m_regAlignTrigger->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMEmMeasurement::handleExperimentLoadChanged()
{
    debugLog() << "handleExperimentLoadChanged() called!";

    bool rc = false;
    m_expLoadStatus->set( LOADING_DATA, true );

    m_subject.reset( new LaBP::WDataSetEMMSubject() );

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

const std::string WMEmMeasurement::NO_DATA_LOADED = "No data loaded.";
const std::string WMEmMeasurement::LOADING_DATA = "Loading data ...";
const std::string WMEmMeasurement::DATA_LOADED = "Data successfully loaded.";
const std::string WMEmMeasurement::DATA_ERROR = "Could not load data.";

const std::string WMEmMeasurement::NO_FILE_LOADED = "No file loaded.";
const std::string WMEmMeasurement::LOADING_FILE = "Loading file ...";
const std::string WMEmMeasurement::FILE_LOADED = "File successfully loaded.";
const std::string WMEmMeasurement::FILE_ERROR = "Could not load file.";
