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

#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WMPCA.h"
#include "WMPCA.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMPCA )

WMPCA::WMPCA()
{
}

WMPCA::~WMPCA()
{
}

boost::shared_ptr< WModule > WMPCA::factory() const
{
    return boost::shared_ptr< WModule >( new WMPCA() );
}

const char** WMPCA::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMPCA::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " PCA";
}

const std::string WMPCA::getDescription() const
{
    return "Employs PCA to reduce the dimensionality of EEG/MEG data."; // TODO(kaehler): Comments
}

void WMPCA::connectors()
{
    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 32, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr(
                    new WLModuleOutputDataCollectionable< WLEMMCommand >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMPCA::properties()
{
    WLModuleDrawable::properties();

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    m_propGrpPCAComputation = m_properties->addPropertyGroup( "PCA Computation Properties",
                    "Contains properties for the PCA algorithm", false );

    //TODO: retrieve the maximum dimension parameter from the EEG/MEG data
    //TODO: allow an minimum accuracy to be specified instead of the # of dimensions

    // the frequencies
    //m_finalDimensions = m_propGrpPCAComputation->addProperty( "Final Dimensionality",
    //                "Set the number of dimensions for the reduced data.", 5.0 );

    m_finalDimensions = m_propGrpPCAComputation->addProperty( "Average Type", "Choose a average type.", 5,
                    boost::bind( &WMPCA::callbackPCATypeChanged, this ) );
    m_finalDimensions->setMin( 1 );
    m_finalDimensions->setMax( 100 );

    m_reverse = m_propGrpPCAComputation->addProperty( "Reverse", "Should the data be presented in the original basis?", false,
                    boost::bind( &WMPCA::callbackPCATypeChanged, this ) );

    m_processModality = boost::shared_ptr< WItemSelection >( new WItemSelection() );
    std::set< WLEModality::Enum > mEnums = WLEModality::values();
    for( std::set< WLEModality::Enum >::iterator it = mEnums.begin(); it != mEnums.end(); ++it )
    {
        m_processModality->addItem(
                        boost::shared_ptr< WItemSelectionItemTyped< WLEModality::Enum > >(
                                        new WItemSelectionItemTyped< WLEModality::Enum >( *it, WLEModality::name( *it ),
                                                        WLEModality::name( *it ) ) ) );
    }

    // getting the SelectorProperty from the list an add it to the properties
    m_processModalitySelection = m_propGrpPCAComputation->addProperty( "Process Modality",
                    "What kind of filter do you want to use", m_processModality->getSelectorFirst(),
                    boost::bind( &WMPCA::callbackProcessModalityChanged, this ) );

    // Be sure it is at least one selected, but not more than one
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_processModalitySelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_processModalitySelection );
}

void WMPCA::moduleInit()
{
    infoLog() << "Initializing module ...";
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::DYNAMIC ); // TODO(pieloth): No 3D needed!
    m_pca.reset( new WPCA( m_finalDimensions->get(), m_reverse->get() ) );
    infoLog() << "Initializing module finished!";
}

void WMPCA::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    // m_moduleState.add( m_propUseCuda->getCondition() ); // when useCuda changed
    m_moduleState.add( m_propCondition ); // when properties changed

    WLEMMCommand::SPtr labpIn;

    ready(); // signal ready state

    moduleInit();

    debugLog() << "Entering main loop";
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() )
        {
            debugLog() << "Waiting for Events";
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        labpIn.reset();
        if( !m_input->isEmpty() )
        {
            labpIn = m_input->getData();
        }
        const bool dataValid = ( labpIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid && labpIn->hasEmm() ) // If there was an update on the inputconnector
        {
            WLEMMeasurement::SPtr emmIn;
            debugLog() << "received data";
            // TODO(pieloth): new profiler

            //m_pca->setParams( m_finalDimensions->get(), m_reverse->get() );
            WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *emmIn ) );

//            for( size_t mod = 0; mod < m_emm->getModalityCount(); ++mod )
//            {
            WLEModality::Enum mod =
                            m_processModalitySelection->get().at( 0 )->getAs< WItemSelectionItemTyped< WLEModality::Enum > >()->getValue();
            WLEMData::SPtr emdIn = emmIn->getModality( mod );
            WLEMData::SPtr emdOut = m_pca->processData( emdIn );

            // emm object can create in the outer module
            emmOut->addModality( emdOut );
//            }

            WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
            labp->setEmm( emmOut );
            m_output->updateData( labp );
            viewUpdate( emmOut );
        }
    }
}

void WMPCA::callbackPCATypeChanged()
{
    debugLog() << "handlePCATypeChanged() called!";
    m_pca->setParams( m_finalDimensions->get(), m_reverse->get() );
}

void WMPCA::callbackProcessModalityChanged( void )
{
    debugLog() << "handleProcessModalityChanged() called!";
}

bool WMPCA::processCompute( WLEMMeasurement::SPtr emm )
{
    // TODO(pieloth): use method for computation
    WLEMMCommand::SPtr labp( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    labp->setEmm( emm );
    m_output->updateData( labp );
    return true;
}

bool WMPCA::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMPCA::processReset( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}
