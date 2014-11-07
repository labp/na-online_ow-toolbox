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
#include "core/io/WLReaderMAT.h"
#include "core/module/WLConstantsModule.h"

#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/io/WLWriterMAT.h"

//include Beamforming files
#include "WBeamforming.h"
#include "WBeamformingCPU.h"
#ifdef FOUND_CUDA
#include "WBeamformingCuda.h"
#endif
#include "WMBeamforming.xpm"
#include "WMBeamforming.h"

using std::set;
using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMBeamforming )

// Leadfield file status
static const std::string NO_MATRIX_LOADED = "No matrix loaded.";
static const std::string LOADING_MATRIX = "Loading matrix ...";
static const std::string MATRIX_LOADED = "Matrix successfully loaded.";

//*************************TEST************************************
// static const std::string DATA_NOT_LOADED = "No data loaded.";
// static const std::string DATA_LOADING = "Loading data ...";
// static const std::string DATA_LOADED = "Data successfully loaded.";
//**********************************************************************

WMBeamforming::WMBeamforming()
{
    m_lastModality = WLEModality::UNKNOWN;
}

WMBeamforming::~WMBeamforming()
{
}

WModule::SPtr WMBeamforming::factory() const
{
    return WModule::SPtr( new WMBeamforming() );
}

const char** WMBeamforming::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMBeamforming::getName() const
{
    return WLConstantsModule::generateModuleName( "Beamforming" );
}

const std::string WMBeamforming::getDescription() const
{
    return "A general signal processing technique used to control the directionality of the reception "
                    " or transmission of a signal on a transducer array.";
}

void WMBeamforming::connectors()
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
//    WLModuleDrawable::properties();
//    WLModuleDrawable::setTimerangeInformationOnly( true );
//    WLModuleDrawable::setViewModality( WLEModality::SOURCE );
//    WLModuleDrawable::hideViewModalitySelection( true );
//    WLModuleDrawable::hideLabelChanged( true );
//    WLModuleDrawable::setComputeModalitySelection( WLEModality::valuesLocalizeable() );
//    WLModuleDrawable::hideComputeModalitySelection( true );
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::setViewModality( WLEModality::SOURCE );
    WLModuleDrawable::hideViewModalitySelection( true );
    WLModuleDrawable::hideLabelChanged( true );
    WLModuleDrawable::setComputeModalitySelection( WLEModality::valuesLocalizeable() );

    // wait- function
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );                      //muss
    // Structure properties with group
    m_propGrpBeamforming = m_properties->addPropertyGroup( "Beamforming",                       //beschreibung
                    "Contains properties for Beamforming.", false );
    m_type = WItemSelection::SPtr( new WItemSelection() );
    std::set< WBeamforming::WEType::Enum > wEnums = WBeamforming::WEType::values();
    for( std::set< WBeamforming::WEType::Enum >::iterator it = wEnums.begin(); it != wEnums.end(); ++it )
    {
        m_type->addItem(
                        WItemSelectionItemTyped< WBeamforming::WEType::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WBeamforming::WEType::Enum >( *it,
                                                        WBeamforming::WEType::name( *it ) ) ) );
    }

    m_typeSelection = m_propGrpBeamforming->addProperty( "Type", "What kind of beamformer do you want to use",
                    m_type->getSelectorFirst(), m_propCondition );

    // covariance/csd
    m_CSDFile = m_propGrpBeamforming->addProperty( "csd or cov file:",
                    "Read a MAT file containing the cross spectral density or covariance.", WPathHelper::getHomePath(),
                    m_propCondition );                              //Button zum Leadfield einlesen
    m_CSDFile->changed( true );
    m_CSDStatus = m_propGrpBeamforming->addProperty( "csd or cov file status:", "csd or cov file status.", NO_MATRIX_LOADED );
    m_CSDStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_reg = m_propGrpBeamforming->addProperty( "reg", "Value for regularization LCMV. For reg=0 -> without regularization.", 0.0,
                    m_propCondition );

    // CUDA properties
    m_useCuda = m_propGrpBeamforming->addProperty( "Use Cuda", "Activate CUDA support.", true, m_propCondition );
    m_useCuda->changed( true );
#ifndef FOUND_CUDA
    m_useCuda->setHidden( true );
#endif // FOUND_CUDA
}

void WMBeamforming::moduleInit()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true );                   // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() );    // when inputdata changed
    m_moduleState.add( m_propCondition );                       // when properties changed

    ready();                                                        // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );               //view 2D

    infoLog() << "Restoring module ...";

    // csd/covariance
    if( m_CSDFile->changed( true ) )
    {
        handleCSDChanged( m_CSDFile->get().string(), &m_CSD );
    }

    handleImplementationChanged();

    infoLog() << "Restoring module finished!";
}

void WMBeamforming::moduleMain()
{
    moduleInit();
    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() )                                         // continue processing if data is available
        {
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
        // csd/covariance
        if( m_CSDFile->changed( true ) )                                                          // Leadfield File MATLAB
        {
            handleCSDChanged( m_CSDFile->get().string(), &m_CSD );
        }

        if( m_lastModality != getCalculateModality() )
        {
            handleComputeModalityChanged();
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );
        if( dataValid )
        {
            process( cmdIn );
        }
    }

    viewCleanup();
}

void WMBeamforming::handleImplementationChanged( void )
{
    debugLog() << __func__ << "() called!";

    if( m_useCuda->get() )
    {
#ifdef FOUND_CUDA
        infoLog() << "Using Beamforming for CUDA.";
        m_beamforming = WBeamformingCuda::SPtr( new WBeamformingCuda() );
#else
        errorLog() << "Build process has detected, that your machine has no CUDA support! Using CPU instead.";
        m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
#endif // FOUND_CUDA
    }
    else
    {
        infoLog() << "Using SourceReconstruction for CPU.";
        m_beamforming = WBeamformingCPU::SPtr( new WBeamformingCPU() );
    }
}

void WMBeamforming::handleComputeModalityChanged()
{
    debugLog() << __func__ << "() called!";

    m_lastModality = getCalculateModality();
    m_beamforming->reset();
}

bool WMBeamforming::handleCSDChanged( std::string fName, Eigen::MatrixXcd* const csd )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Reading MAT" ) );
    m_progress->addSubProgress( progress );
    m_CSDStatus->set( LOADING_MATRIX, true );

    WLReaderMAT::SPtr reader; // Matlab
    try
    {
        reader.reset( new WLReaderMAT( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    if( reader->read( csd ) == WLIOStatus::SUCCESS )
    {
        debugLog() << "read file ";
        m_CSDStatus->set( MATRIX_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );

        return true;
    }
    else
    {
        errorLog() << "Could not read leadfield!";
        m_CSDStatus->set( NO_MATRIX_LOADED, true );
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
    m_beamforming->setType(
                    m_typeSelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WBeamforming::WEType::Enum > >()->getValue() );

    WLEModality::Enum modality = this->getCalculateModality();
    WLEMData::SPtr eeg = emmIn->getModality( modality );
    if( !m_beamforming->hasBeam() )
    {
        if( !emmIn->getSubject()->hasLeadfield( modality ) )
        {
            errorLog() << "No leadfield available!";
            return false;
        }
        WLMatrix::ConstSPtr lf = emmIn->getSubject()->getLeadfield( modality );
        if( !m_beamforming->calculateBeamforming( *lf, m_CSD, m_reg->get() ) )
        {
            errorLog() << "Could not prepare beamformer!";
            return false;
        }
    }

    WLEMDSource::SPtr sourceOut;
    try
    {
        sourceOut = m_beamforming->beam( emmIn->getModality( modality ) );
    }
    catch( const std::exception& e )
    {
        errorLog() << e.what();
        return false;
    }

    WLEMMeasurement::SPtr emmOut = emmIn->clone();
    emmOut->addModality( emmIn->getModality( modality ) );
    emmOut->addModality( sourceOut );
    debugLog() << "Matrix: " << sourceOut->getNrChans() << " x " << sourceOut->getSamplesPerChan();

    viewUpdate( emmOut );

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
    return true;
}
