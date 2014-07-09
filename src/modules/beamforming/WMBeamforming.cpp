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

#include <exception>
#include <string>
#include <set>

#include <boost/shared_ptr.hpp>

#include <core/common/WItemSelection.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPropertyHelper.h>
#include <core/common/WPathHelper.h>
#include <core/kernel/WModule.h>

// Input & output data
#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/data/enum/WLEModality.h"
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderMAT.h"
#include "core/module/WLConstantsModule.h"
#ifdef FOUND_CUDA
#include "WBeamformingCuda.h"
#endif
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/io/WLWriterMAT.h"
//include Beamforming files
#include "WBeamforming.h"
#include "WBeamformingCPU.h"
#include "beam.xpm"
#include "WMBeamforming.h"
#include <string>
using std::set;
using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMBeamforming )
int i;
// Leadfield file status
static const std::string NO_MATRIX_LOADED = "No matrix loaded.";
static const std::string LOADING_MATRIX = "Loading matrix ...";
static const std::string MATRIX_LOADED = "Matrix successfully loaded.";
/*

//*************************TEST************************************
static const std::string DATA_NOT_LOADED = "No data loaded.";
static const std::string DATA_LOADING = "Loading data ...";
static const std::string DATA_LOADED = "Data successfully loaded.";
//**********************************************************************
*/
WMBeamforming::WMBeamforming()
{
  m_lastModality = WLEModality::UNKNOWN;

}

WMBeamforming::~WMBeamforming()
{
}

WModule::SPtr WMBeamforming::factory() const
{
    return WModule::SPtr( new WMBeamforming() );            //new module
}

const char** WMBeamforming::getXPMIcon() const
{
    return XPM_Beam;                                //Bezeichnung in xpm-Datei
}

const std::string WMBeamforming::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Beamforming"; //Box : Name NA-ONline: Beamforming
}

const std::string WMBeamforming::getDescription() const
{

    return "A general signal processing technique used to control the directionality of the reception "
                    " or transmission of a signal on a transducer array.";
}

void WMBeamforming::connectors()        //anschlüsse der Modul-Box , ein- und ausgang
{
    WLModuleDrawable::connectors();
    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}



void WMBeamforming::properties()
{   //View-Fenster
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::setViewModality( WLEModality::SOURCE );
    WLModuleDrawable::hideViewModalitySelection( true );
    WLModuleDrawable::hideLabelChanged( true );
    WLModuleDrawable::setComputeModalitySelection( WLEModality::valuesLocalizeable() );
    WLModuleDrawable::hideComputeModalitySelection( true );

    //wait- function
        m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );                      //muss
    //Structure properties with group
        m_propGrpBeamforming = m_properties->addPropertyGroup( "Beamforming",                       //beschreibung
                        "Contains properties for Beamforming.", false );

    // Algorithm reset //
        m_resetModule = m_propGrpBeamforming->addProperty( "Reset module:", "Re(set)", WPVBaseTypes::PV_TRIGGER_READY,      //nötig?
                        m_propCondition );

//    /*//read Leadfield,change, properties in control panel
//        m_lfMEGFile = m_propGrpBeamforming->addProperty( "Leadfield MEG file:", "Read a FIFF file containing the leadfield for MEG.",
//                        WPathHelper::getHomePath(), m_propCondition );                              //Button zum Leadfield einlesen
//        m_lfMEGFile->changed( true ); */                                                              //select Leadfield
//        //read Leadfield,change, properties in control panel
//

                       m_lfEEGFile = m_propGrpBeamforming->addProperty( "Leadfield EEG file:", "Read a FIFF file containing the leadfield for EEG.",
                                       WPathHelper::getHomePath(), m_propCondition );                              //Button zum Leadfield einlesen
                       m_lfEEGFile->changed( true );
           // Leadfield properties,display, properties in control panel
               m_leadfieldStatus = m_propGrpBeamforming->addProperty( "Leadfield file status:", "Leadfield file status.", NO_MATRIX_LOADED );
               m_leadfieldStatus->setPurpose( PV_PURPOSE_INFORMATION );

///////////////////////


               m_NoiseFile = m_propGrpBeamforming->addProperty( "Noise EEG file:", "Read a MAT file containing the leadfield for EEG.",
                                                     WPathHelper::getHomePath(), m_propCondition );                              //Button zum Leadfield einlesen
               m_NoiseFile->changed( true );

               m_NoiseStatus= m_propGrpBeamforming->addProperty( "Noise file status:", "Noise file status.", NO_MATRIX_LOADED );
               m_NoiseStatus->setPurpose( PV_PURPOSE_INFORMATION );

               m_DataFile = m_propGrpBeamforming->addProperty( "Data EEG file:", "Read a MAT file containing the leadfield for EEG.",
                                                                  WPathHelper::getHomePath(), m_propCondition );                              //Button zum Leadfield einlesen
                            m_DataFile->changed( true );

                            m_DataStatus= m_propGrpBeamforming->addProperty( "Data file status:", "Data file status.", NO_MATRIX_LOADED );
                            m_DataStatus->setPurpose( PV_PURPOSE_INFORMATION );

/////////////////////////////////

        m_useCuda = m_propGrpBeamforming->addProperty( "Use Cuda", "Activate CUDA support.", true, m_propCondition );
            m_useCuda->changed( true );
        #ifndef FOUND_CUDA
            m_useCuda->setHidden( true );
        #endif // FOUND_CUDA

}

void WMBeamforming::moduleInit()
{
    infoLog() << "Initializing module ...";


    // init moduleState for using Events in mainLoop
        m_moduleState.setResetable( true, true );                   // resetable, autoreset
        m_moduleState.add( m_input->getDataChangedCondition() );    // when inputdata changed
        m_moduleState.add( m_propCondition );                       // when properties changed
        m_subject.reset( new WLEMMSubject() );                      //Leadfield
//   m_beamforming.reset( new WBeamforming() );                     //ohne CPU und ohne =0 bei beam()
        m_leadfieldEEG.reset();

    ready();                                                        // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );               //view 2D

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    infoLog() << "Restoring module finished!";


        if( m_lfEEGFile->changed( true ) )                                                              //Leadfield  File MATLAB
        {

            if( handleLfFileChanged( m_lfEEGFile->get().string(), m_leadfieldEEG ) )
            {
                m_subject->setLeadfield( WLEModality::EEG, m_leadfieldEEG );
            }
        }
        //////////////////////////////////////////
        if( m_NoiseFile->changed( true ) )                                                          //Leadfield File MATLAB
                {
                    if( handleNoiseChanged( m_NoiseFile->get().string(), m_Noise ) )
                    {
                        m_subject->setLeadfield( WLEModality::EEG, m_Noise);
                    }
                }
        //////////////////////////////////////////
         if( m_DataFile->changed( true ) )                                                          //Leadfield File MATLAB
                 {
                     if( handleNoiseChanged( m_DataFile->get().string(), m_Data ) )
                     {
                         m_subject->setLeadfield( WLEModality::EEG, m_Data);
                     }
                 }
///////////////////////////////

    handleImplementationChanged();
    infoLog() << "Restoring module finished!";

}

void WMBeamforming::moduleMain()
{    debugLog() << "modul main ";
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() )                                         // continue processing if data is available
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait();                                       // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break;                                                      // break mainLoop on shutdown
        }
        if( m_useCuda->changed( true ) )
             {
                 handleImplementationChanged();
             }

        if( m_resetModule->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleResetTrigger();
        }

       cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();

        }

        if( m_lastModality != getCalculateModality() )
        {
            handleComputeModalityChanged( cmdIn );
        }

        if( m_lfEEGFile->changed( true ) )                                                          //Leadfield File MATLAB
        {
            if( handleLfFileChanged( m_lfEEGFile->get().string(), m_leadfieldEEG ) )
            {
                m_subject->setLeadfield( WLEModality::EEG, m_leadfieldEEG );
            }
        }
        //////////////////////////////////////////
        if( m_NoiseFile->changed( true ) )                                                          //Leadfield File MATLAB
                {
                    if( handleNoiseChanged( m_NoiseFile->get().string(), m_Noise ) )
                    {
                        m_subject->setLeadfield( WLEModality::EEG, m_Noise);
                    }
                }
        /////////////////////
        if( m_DataFile->changed( true ) )                                                          //Leadfield File MATLAB
                  {
                      if( handleNoiseChanged( m_DataFile->get().string(), m_Data ) )
                      {
                          m_subject->setLeadfield( WLEModality::EEG, m_Data);
                      }
                  }
///////////////////////////////
        const bool dataValid = ( cmdIn);

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid )                                                      // If there was an update on the inputconnector
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}
//void  WMBeamforming::handleImplementationChanged( void )
//{
//    debugLog() << "callbackImplementationChanged() called!";
//
//    if( m_useCuda->get() )
//    {
//#ifdef FOUND_CUDA
//        infoLog() << "Using Beamforming for CUDA.";
//        m_beamforming = WBeamformingCuda::SPtr( new WBeamformingCuda() );
//#else
//        errorLog() << "Build process has detected, that your machine has no CUDA support! Using CPU instead.";
//        m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
//#endif // FOUND_CUDA
//    }
//    else
//    {
//        infoLog() << "Using SourceReconstruction for CPU.";
//        m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
//    }
//}
void WMBeamforming::handleResetTrigger()
{
    debugLog() << "handleResetTrigger() called!";

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
    processReset( cmd );
    i=0;

    m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY );


}

void WMBeamforming::handleImplementationChanged( void )
{
    m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
}


void WMBeamforming::handleComputeModalityChanged( WLEMMCommand::ConstSPtr cmd )
{
    debugLog() << "handleComputeModalityChanged()";

    m_lastModality = getCalculateModality();
    m_beamforming->reset();
}

bool WMBeamforming::handleLfFileChanged( std::string fName, WLMatrix::SPtr& lf )            //Read Leadfield FIFF File
{
    debugLog() << "handleLfFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading Leadfield" ) );
    m_progress->addSubProgress( progress );
    m_leadfieldStatus->set( LOADING_MATRIX, true );

// WLReaderLeadfield::SPtr reader;
 WLReaderMAT::SPtr reader; //Matlab
    try
    {
     //reader.reset( new WLReaderLeadfield( fName ) );
        reader.reset( new WLReaderMAT( fName ) );  //Matlab
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }


   // if( reader->read( lf ) == WLIOStatus::SUCCESS )
   if( reader->readMatrix( lf ) == WLIOStatus::SUCCESS ) //Matlab

    {    debugLog() << "read file ";
        m_leadfieldStatus->set( MATRIX_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );

        return true;
    }

    else
    {
        errorLog() << "Could not read leadfield!";
        m_leadfieldStatus->set( NO_MATRIX_LOADED, true );
        progress->finish();


        m_progress->removeSubProgress( progress );
        return false;
    }


}

///////////////////////////////////////////////////////////////////////

bool WMBeamforming::handleNoiseChanged( std::string fName, WLMatrix::SPtr& data )            //Read Leadfield FIFF File
{
    debugLog() << "handleLfFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading MAT" ) );
    m_progress->addSubProgress( progress );
    m_NoiseStatus->set( LOADING_MATRIX, true );

 //WLReaderLeadfield::SPtr reader;
  WLReaderMAT::SPtr reader; //Matlab
    try
    {
     //   reader.reset( new WLReaderLeadfield( fName ) );
        reader.reset( new WLReaderMAT( fName ) );  //Matlab
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }


    //if( reader->read( lf ) == WLIOStatus::SUCCESS )
   if( reader->readMatrix( data ) == WLIOStatus::SUCCESS ) //Matlab

    {    debugLog() << "read file ";
    m_NoiseStatus->set( MATRIX_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );

        return true;
    }

    else
    {
        errorLog() << "Could not read leadfield!";
        m_NoiseStatus->set( NO_MATRIX_LOADED, true );
        progress->finish();


        m_progress->removeSubProgress( progress );
        return false;
    }


}
bool WMBeamforming::handleDataChanged( std::string fName, WLMatrix::SPtr& data )            //Read Leadfield FIFF File
{
    debugLog() << "handleLfFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading MAT" ) );
    m_progress->addSubProgress( progress );
    m_DataStatus->set( LOADING_MATRIX, true );

 //WLReaderLeadfield::SPtr reader;
  WLReaderMAT::SPtr reader; //Matlab
    try
    {
     //   reader.reset( new WLReaderLeadfield( fName ) );
        reader.reset( new WLReaderMAT( fName ) );  //Matlab
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }


    //if( reader->read( lf ) == WLIOStatus::SUCCESS )
   if( reader->readMatrix( data ) == WLIOStatus::SUCCESS ) //Matlab

    {    debugLog() << "read file ";
    m_DataStatus->set( MATRIX_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );

        return true;
    }

    else
    {
        errorLog() << "Could not read leadfield!";
        m_DataStatus->set( NO_MATRIX_LOADED, true );
        progress->finish();


        m_progress->removeSubProgress( progress );
        return false;
    }


}

bool WMBeamforming::processInit( WLEMMCommand::SPtr cmdIn )
{
    m_output->updateData( cmdIn );
  return true;
}
bool WMBeamforming::processCompute( WLEMMeasurement::SPtr emmIn )
{

    // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
        debugLog() << "received data";


    // hole EEG, püfe ob EEG vohanden
        WLEMData::SPtr eeg = emmIn->getModality( WLEModality::EEG );


        m_beamforming->calculateBeamforming( eeg->getData(), *m_leadfieldEEG ,*m_Noise,*m_Data);// DATEN IN FUNKTION
       // m_beamforming->calculateBeamforming( eeg->getData(), *m_leadfieldEEG );


        WLEMMeasurement::SPtr emmOut;

        // erstelle sourceOut und setze Daten
             WLEMDSource::SPtr sourceOut;
            try
            {
                sourceOut = m_beamforming->beam( emmIn->getModality( WLEModality::EEG  ) );
       /*         MatrixT Source;
                Source= *sourceOut;
                i++;


                std::string String = static_cast<std::ostringstream*>( &(std::ostringstream() << i) )->str();
                WLEMData::DataSPtr GG(new WLEMData::DataT (*sourceOut));
                WLMatrix::ConstSPtr RR( new  WLMatrix::ConstSPtr(GG) );

                WLWriterMAT::writeMatrix( RR ,String);*/

            }
            catch( const std::exception& e )
            {
                errorLog() << e.what();
                return false;
            }
//
        // erstelle emmOut ... bzw. clone/new ...
            emmOut = emmIn->clone();        //Quellenanzahl x Abtastwerte
            emmOut->addModality( emmIn->getModality(  WLEModality::EEG  ) );
        // setze in emmOut deine Daten addMod(emdSource)
            emmOut->addModality( sourceOut);
            infoLog() << "Matrix: " << sourceOut->getNrChans() << " x " << sourceOut->getSamplesPerChan();

        viewUpdate( emmOut );                                                                                //IMMER -bis true

        WLEMMCommand::SPtr cmdOut = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
        cmdOut->setEmm( emmOut );
        m_output->updateData( cmdOut );

        return true;


}




bool WMBeamforming::processReset( WLEMMCommand::SPtr cmdIn )
{
    m_input->clear();
    viewReset();
    m_beamforming->reset();
    m_output->updateData( cmdIn );

    //get/set col Leadfield
//        int value = m_source->get();
//        m_beamforming->setSource( value );

    return true;

}

