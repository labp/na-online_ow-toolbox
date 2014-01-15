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
#include <core/kernel/WModule.h>

// Input & output data
#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
// Input & output connectors
// TODO(pieloth): use OW classes
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/util/profiler/WLTimeProfiler.h"

#include "WSourceReconstruction.h"
#include "WSourceReconstructionCpu.h"
#ifdef FOUND_CUDA
#include "WSourceReconstructionCuda.h"
#endif

#include "WMSourceReconstruction.h"
#include "WMSourceReconstruction.xpm"

using std::set;
using WLMatrix::MatrixT;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMSourceReconstruction )

WMSourceReconstruction::WMSourceReconstruction()
{
}

WMSourceReconstruction::~WMSourceReconstruction()
{
}

WModule::SPtr WMSourceReconstruction::factory() const
{
    return WModule::SPtr( new WMSourceReconstruction() );
}

const char** WMSourceReconstruction::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMSourceReconstruction::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Source Reconstruction";
}

const std::string WMSourceReconstruction::getDescription() const
{
    return "Estimates a source distribution according to a linear source reconstruction algorithm.";
}

void WMSourceReconstruction::connectors()
{
    m_input = LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMSourceReconstruction::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::setViewModality( WLEModality::SOURCE );
    WLModuleDrawable::hideViewModalitySelection( true );
    WLModuleDrawable::hideLabelChanged( true );
    WLModuleDrawable::setComputeModalitySelection( WLEModality::valuesLocalizeable() );

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    m_propGrpSourceReconstruction = m_properties->addPropertyGroup( "Source Reconstruction",
                    "Contains properties for Source Reconstruction.", false );

    m_useCuda = m_propGrpSourceReconstruction->addProperty( "Use Cuda", "Activate CUDA support.", true, m_propCondition );
    m_useCuda->changed( true );
#ifndef FOUND_CUDA
    m_useCuda->setHidden( true );
#endif // FOUND_CUDA
    // Algorithm reset //
    m_resetModule = m_propGrpSourceReconstruction->addProperty( "Reset module:", "Reset", WPVBaseTypes::PV_TRIGGER_READY,
                    m_propCondition );

    // Leadfield properties
    m_leadfieldStatus = m_propGrpSourceReconstruction->addProperty( "Leadfield file status:", "Leadfield file status.",
                    NO_MATRIX_LOADED );
    m_leadfieldStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_leadfieldRows = m_propGrpSourceReconstruction->addProperty( "LF rows:", "Rows of leadfield matrix.", 0 );
    m_leadfieldRows->setPurpose( PV_PURPOSE_INFORMATION );

    m_leadfieldCols = m_propGrpSourceReconstruction->addProperty( "LF columns:", "columns of leadfield matrix.", 0 );
    m_leadfieldCols->setPurpose( PV_PURPOSE_INFORMATION );

    // Weighting properties
    m_weightingStatus = m_propGrpSourceReconstruction->addProperty( "Weighting file status:", "Weighting file status.",
                    NO_MATRIX_LOADED );
    m_weightingStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_weightingTypes = WItemSelection::SPtr( new WItemSelection() );
    std::set< WSourceReconstruction::WEWeightingCalculation::Enum > wEnums =
                    WSourceReconstruction::WEWeightingCalculation::values();
    for( std::set< WSourceReconstruction::WEWeightingCalculation::Enum >::iterator it = wEnums.begin(); it != wEnums.end(); ++it )
    {
        m_weightingTypes->addItem(
                        WItemSelectionItemTyped< WSourceReconstruction::WEWeightingCalculation::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WSourceReconstruction::WEWeightingCalculation::Enum >( *it,
                                                        WSourceReconstruction::WEWeightingCalculation::name( *it ),
                                                        WSourceReconstruction::WEWeightingCalculation::name( *it ) ) ) );
    }

    // getting the SelectorProperty from the list an add it to the properties
    m_weightingTypesSelection = m_propGrpSourceReconstruction->addProperty( "Weighting Type", "Choose a weighting norm.",
                    m_weightingTypes->getSelectorFirst(), m_propCondition );
    m_weightingTypesSelection->changed( true );

    // Be sure it is at least one selected, but not more than one
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_weightingTypesSelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_weightingTypesSelection );

    m_weightingRows = m_propGrpSourceReconstruction->addProperty( "Weighting rows:", "Rows of weighting matrix.", 0 );
    m_weightingRows->setPurpose( PV_PURPOSE_INFORMATION );

    m_weightingCols = m_propGrpSourceReconstruction->addProperty( "Weighting columns:", "columns of weighting matrix.", 0 );
    m_weightingCols->setPurpose( PV_PURPOSE_INFORMATION );

    m_snr = m_propGrpSourceReconstruction->addProperty( "SNR", "SNR value for inverse solution.", 20.0, m_propCondition );
    m_snr->changed( true );

    // inverse solution
    m_inverseStatus = m_propGrpSourceReconstruction->addProperty( "Inverse status:", "Inverse solution status.",
                    NO_MATRIX_LOADED );
    m_inverseStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_inverseRows = m_propGrpSourceReconstruction->addProperty( "Inverse rows:", "Rows of inverse matrix.", 0 );
    m_inverseRows->setPurpose( PV_PURPOSE_INFORMATION );

    m_inverseCols = m_propGrpSourceReconstruction->addProperty( "Inverse columns:", "columns of inverse matrix.", 0 );
    m_inverseCols->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMSourceReconstruction::moduleInit()
{
    infoLog() << "Initializing module ...";

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( LaBP::WLEMDDrawable2D::WEGraphType::SINGLE );

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    handleImplementationChanged();
    handleWeightingTypeChanged();
    handleSnrChanged();

    infoLog() << "Restoring module finished!";
}

void WMSourceReconstruction::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmd;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        if( m_resetModule->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            handleResetTrigger();
        }

        if( m_weightingTypesSelection->changed( true ) )
        {
            handleWeightingTypeChanged();
        }

        if( m_snr->changed( true ) )
        {
            handleSnrChanged();
        }

        if( m_useCuda->changed( true ) )
        {
            handleImplementationChanged();
        }

        cmd.reset();
        if( !m_input->isEmpty() )
        {
            cmd = m_input->getData();
        }

        if( m_lastModality != getCalculateModality() )
        {
            handleComputeModalityChanged( cmd );
        }

        const bool dataValid = ( cmd );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( cmd );
        }
    }
}

void WMSourceReconstruction::handleImplementationChanged( void )
{
    debugLog() << "callbackImplementationChanged() called!";

    if( m_useCuda->get() )
    {
#ifdef FOUND_CUDA
        infoLog() << "Using SourceReconstruction for CUDA.";
        m_sourceReconstruction = WSourceReconstructionCuda::SPtr( new WSourceReconstructionCuda() );
#else
        errorLog() << "Build process has detected, that your machine has no CUDA support! Using CPU instead.";
        m_sourceReconstruction = WSourceReconstructionCpu::SPtr( new WSourceReconstructionCpu() );
#endif // FOUND_CUDA
    }
    else
    {
        infoLog() << "Using SourceReconstruction for CPU.";
        m_sourceReconstruction = WSourceReconstructionCpu::SPtr( new WSourceReconstructionCpu() );
    }
}

void WMSourceReconstruction::handleResetTrigger()
{
    debugLog() << "handleResetTrigger() called!";

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
    processReset( cmd );

    m_resetModule->set( WPVBaseTypes::PV_TRIGGER_READY );
}

void WMSourceReconstruction::handleWeightingTypeChanged()
{
    debugLog() << "handleWeightingTypeChanged() called!";

    m_weightingStatus->set( WMSourceReconstruction::LOADING_MATRIX, true );
    WSourceReconstruction::WEWeightingCalculation::Enum type = m_weightingTypesSelection->get().at( 0 )->getAs<
                    WItemSelectionItemTyped< WSourceReconstruction::WEWeightingCalculation::Enum > >()->getValue();
    if( m_sourceReconstruction->calculateWeightningMatrix( type ) )
    {
        m_weightingRows->set( m_sourceReconstruction->getWeighting().rows(), true );
        m_weightingCols->set( m_sourceReconstruction->getWeighting().cols(), true );
        m_weightingStatus->set( WMSourceReconstruction::MATRIX_LOADED, true );

        infoLog() << "Weighting matrix calculated!";
    }
    else
    {
        m_weightingRows->set( 0, true );
        m_weightingCols->set( 0, true );
        m_weightingStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );
    }
}

void WMSourceReconstruction::handleSnrChanged()
{
    WLTimeProfiler tp( "WMSourceReconstruction", "handleSnrChanged" );
    debugLog() << "handleSnrChanged() called!";
    if( !m_nCovarianceMatrix || !m_dCovarianceMatrix )
    {
        errorLog() << "No data or noise covariance matrix available!";
        return;
    }

    m_inverseStatus->set( WMSourceReconstruction::LOADING_MATRIX, true );
    if( m_sourceReconstruction->calculateInverseSolution( *m_nCovarianceMatrix, *m_dCovarianceMatrix, m_snr->get() ) )
    {
        m_inverseRows->set( m_sourceReconstruction->getInverse().rows(), true );
        m_inverseCols->set( m_sourceReconstruction->getInverse().cols(), true );
        m_inverseStatus->set( WMSourceReconstruction::MATRIX_LOADED, true );

        infoLog() << "Inverse solution calculated!";
    }
    else
    {
        m_inverseRows->set( 0, true );
        m_inverseCols->set( 0, true );
        m_inverseStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );
    }
}

void WMSourceReconstruction::handleComputeModalityChanged( WLEMMCommand::ConstSPtr cmd )
{
    debugLog() << "handleComputeModalityChanged()";
    m_lastModality = getCalculateModality();
    m_sourceReconstruction->reset();
}

bool WMSourceReconstruction::inverseSolutionFromSubject( WLEMMeasurement::SPtr emm, WLEModality::Enum modality )
{
    WLTimeProfiler tp( "WMSourceReconstruction", "inverseSolutionFromSubject" );
    debugLog() << "inverseSolutionFromSubject() called!";
    WLEMMSubject::SPtr subject = emm->getSubject();
    if( !subject )
    {
        m_leadfieldStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );
        errorLog() << "Can not compute inverse solution without subject!";
        return false;
    }

    WLMatrix::SPtr leadfield;
    try
    {
        leadfield = subject->getLeadfield( modality );
    }
    catch( const WException& ex )
    {
        m_leadfieldStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );
        errorLog() << "No leadfield matrix for modality!";
        return false;
    }

    m_sourceReconstruction->setLeadfield( leadfield );
    m_leadfieldRows->set( leadfield->rows(), true );
    m_leadfieldCols->set( leadfield->cols(), true );
    m_leadfieldStatus->set( WMSourceReconstruction::MATRIX_LOADED, true );

    m_dCovarianceMatrix.reset( new MatrixT( leadfield->rows(), leadfield->rows() ) );
    m_dCovarianceMatrix->setIdentity();

    m_nCovarianceMatrix.reset( new MatrixT( leadfield->rows(), leadfield->rows() ) );
    m_nCovarianceMatrix->setIdentity();

    handleWeightingTypeChanged();
    handleSnrChanged();

    return m_sourceReconstruction->hasInverse();
}

bool WMSourceReconstruction::processCompute( WLEMMeasurement::SPtr emmIn )
{
    WLTimeProfiler tp( "WMSourceReconstruction", "processCompute" );

    WLEMMeasurement::SPtr emmOut;
    WLEMDSource::SPtr sourceOut;
    // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
    debugLog() << "received data";

    WLEModality::Enum modality = this->getCalculateModality();

    if( !m_sourceReconstruction->hasInverse() )
    {
        if( !inverseSolutionFromSubject( emmIn, modality ) )
        {
            errorLog() << "Skip processing due to no inverse solution!";
            return false;
        }
    }

    try
    {
        sourceOut = m_sourceReconstruction->reconstruct( emmIn->getModality( modality ) );
    }
    catch( const std::exception& e )
    {
        errorLog() << e.what();
        return false;
    }
    infoLog() << "Matrix: " << sourceOut->getNrChans() << " x " << sourceOut->getSamplesPerChan();
    // Create output
    emmOut = emmIn->clone();
    for( size_t i = 0; i < emmIn->getModalityCount(); ++i )
    {
        emmOut->addModality( emmIn->getModality( i ) );
    }
    emmOut->addModality( sourceOut );

    viewUpdate( emmOut );

    WLEMMCommand::SPtr labp = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    labp->setEmm( emmOut );
    m_output->updateData( labp );
    return true;
}

bool WMSourceReconstruction::processInit( WLEMMCommand::SPtr cmdIn )
{
    WLTimeProfiler tp( "WMSourceReconstruction", "processInit" );
    WLEModality::Enum modality = this->getCalculateModality();
    if( cmdIn->hasEmm() )
    {
        inverseSolutionFromSubject( cmdIn->getEmm(), modality );
    }

    m_output->updateData( cmdIn );
    return true;
}

bool WMSourceReconstruction::processReset( WLEMMCommand::SPtr cmdIn )
{
    viewReset();
    m_sourceReconstruction->reset();

    m_leadfieldRows->set( 0, true );
    m_leadfieldCols->set( 0, true );
    m_leadfieldStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );

    m_weightingRows->set( 0, true );
    m_weightingCols->set( 0, true );
    m_weightingStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );

    m_inverseRows->set( 0, true );
    m_inverseCols->set( 0, true );
    m_inverseStatus->set( WMSourceReconstruction::NO_MATRIX_LOADED, true );

    m_nCovarianceMatrix.reset();
    m_dCovarianceMatrix.reset();

    m_output->updateData( cmdIn );
    return true;
}

const std::string WMSourceReconstruction::NO_MATRIX_LOADED = "No matrix loaded.";
const std::string WMSourceReconstruction::LOADING_MATRIX = "Loading matrix ...";
const std::string WMSourceReconstruction::MATRIX_LOADED = "Matrix successfully loaded.";
const std::string WMSourceReconstruction::MATRIX_ERROR = "Could not load matrix.";
