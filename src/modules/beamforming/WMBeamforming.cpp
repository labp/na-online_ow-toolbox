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
#include "core/module/WLConstantsModule.h"
#include "core/io/WLReaderBem.h"
#include "core/container/WLList.h"
#include "core/data/WLEMMBemBoundary.h"
//#include "core/data/WLEMMBemBoundary.h"
// Input & output connectors
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

//include Beamforming files
#include "WBeamforming.h"
#include "WBeamformingCPU.h"
#include "beam.xpm"
#include "WMBeamforming.h"

using std::set;
using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMBeamforming )

// Leadfield file status
static const std::string NO_MATRIX_LOADED = "No matrix loaded.";
static const std::string LOADING_MATRIX = "Loading matrix ...";
static const std::string MATRIX_LOADED = "Matrix successfully loaded.";
static const std::string DATA_NOT_LOADED = "No data loaded.";
static const std::string DATA_LOADING = "Loading data ...";
static const std::string DATA_LOADED = "Data successfully loaded.";
static const std::string DATA_ERROR = "Could not load data.";
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

    //read Leadfield,change, properties in control panel
        m_lfMEGFile = m_propGrpBeamforming->addProperty( "Leadfield MEG file:", "Read a FIFF file containing the leadfield for MEG.",
                        WPathHelper::getHomePath(), m_propCondition );                              //Button zum Leadfield einlesen
        m_lfMEGFile->changed( true );                                                               //select Leadfield

    // Leadfield properties,display, properties in control panel
        m_leadfieldStatus = m_propGrpBeamforming->addProperty( "Leadfield file status:", "Leadfield file status.", NO_MATRIX_LOADED );
        m_leadfieldStatus->setPurpose( PV_PURPOSE_INFORMATION );

    //Number of leadfield col
        m_source = m_propGrpBeamforming->addProperty( "Source number:", "The number of leadfield col ", 2 );
        m_source->setMax( 244622 );


        m_bemFile = m_propGrpBeamforming->addProperty( "BEM file:", "Read a FIFF file containing BEM layers.",
                        WPathHelper::getHomePath(), m_propCondition );
        m_bemFile->changed( true );
       m_BemStatus= m_propGrpBeamforming->addProperty( "Bem file status:", "File status.", DATA_NOT_LOADED);
       m_BemStatus->setPurpose( PV_PURPOSE_INFORMATION );


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
        m_leadfieldMEG.reset();

    ready();                                                        // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );               //view 2D

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    infoLog() << "Restoring module finished!";

    if( m_lfMEGFile->changed( true ) )                                                              //Leadfield  File
    {

        if( handleLfFileChanged( m_lfMEGFile->get().string(), m_leadfieldMEG ) )
        {
            m_subject->setLeadfield( WLEModality::MEG, m_leadfieldMEG );
        }
    }
    if( m_bemFile->changed( true ) )
    {
        if( handleBemFileChanged( m_bemFile->get().string() ) )
        {
            m_subject->setBemBoundaries( m_bems );
        }
    }
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

        if( m_lfMEGFile->changed( true ) )                                                          //Leadfield File
        {
            if( handleLfFileChanged( m_lfMEGFile->get().string(), m_leadfieldMEG ) )
            {
                m_subject->setLeadfield( WLEModality::MEG, m_leadfieldMEG );
            }
        }
        if( m_bemFile->changed( true ) )
        {
            if( handleBemFileChanged( m_bemFile->get().string() ) )
            {
                m_subject->setBemBoundaries( m_bems );
            }
        }
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
////    debugLog() << "callbackImplementationChanged() called!";
////
////    if( m_useCuda->get() )
////    {
////#ifdef FOUND_CUDA
////        infoLog() << "Using SourceReconstruction for CUDA.";
////        m_sourceReconstruction = WSourceReconstructionCuda::SPtr( new WSourceReconstructionCuda() );
////#else
////        errorLog() << "Build process has detected, that your machine has no CUDA support! Using CPU instead.";
////        m_sourceReconstruction = WSourceReconstructionCpu::SPtr( new WSourceReconstructionCpu() );
////#endif // FOUND_CUDA
////    }
////    else
////    {
////        infoLog() << "Using SourceReconstruction for CPU.";
////        m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
////    }
//}
void WMBeamforming::handleResetTrigger()
{
    debugLog() << "handleResetTrigger() called!";

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
    processReset( cmd );


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

    WLReaderLeadfield::SPtr reader;
    try
    {
        reader.reset( new WLReaderLeadfield( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }



    if( reader->read( lf ) == WLIOStatus::SUCCESS )
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
bool WMBeamforming::handleBemFileChanged( std::string fName )
{
    debugLog() << "handleBemFileChanged()";

    WProgress::SPtr progress( new WProgress( "Reading BEM Layer" ) );
    m_progress->addSubProgress( progress );
    m_BemStatus->set( DATA_LOADING, true );

    WLReaderBem::SPtr reader;
    try
    {
        reader.reset( new WLReaderBem( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

  m_bems = WLList< WLEMMBemBoundary::SPtr >::instance();
    if( reader->read( m_bems.get() ) )
    {
        infoLog() << "Loaded BEM layer: " << m_bems->size();
        m_BemStatus->set( DATA_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read BEM layers!";
        m_BemStatus->set( DATA_ERROR, true );
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


    // hole MEG, püfe ob MEG vohanden
        WLEMData::SPtr meg = emmIn->getModality( WLEModality::MEG );


        m_beamforming->calculateBeamforming( meg->getData(), *m_leadfieldMEG  );// DATEN IN FUNKTION
        WLEMMeasurement::SPtr emmOut;

        // erstelle sourceOut und setze Daten
             WLEMDSource::SPtr sourceOut;
            try
            {
                sourceOut = m_beamforming->beam( emmIn->getModality( WLEModality::MEG  ) );
            }
            catch( const std::exception& e )
            {
                errorLog() << e.what();
                return false;
            }
        // erstelle emmOut ... bzw. clone/new ...
            emmOut = emmIn->clone();        //Quellenanzahl x Abtastwerte
            emmOut->addModality( emmIn->getModality(  WLEModality::MEG  ) );
        // setze in emmOut deine Daten addMod(emdSource)
            emmOut->addModality( sourceOut);
            infoLog() << "Matrix: " << sourceOut->getNrChans() << " x " << sourceOut->getSamplesPerChan();

        //viewUpdate( emmOut );       //TODO Debug !!                                                                             //IMMER -bis true

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
        int value = m_source->get();
        m_beamforming->setSource( value );

    return true;

}

