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

#include <string>
#include <vector>

#include <boost/bind.hpp>

#include "core/kernel/WModule.h"

// Input & output connectors
#include "core/kernel/WLModuleInputDataRingBuffer.h"
#include "core/kernel/WLModuleOutputDataCollectionable.h"

// Input & output data
#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEEG.h"
#include "core/common/WItemSelectionItemTyped.h"
#include "core/common/WLTimeProfiler.h"

#include "WPCA.h"

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
    return "PCA";
}

const std::string WMPCA::getDescription() const
{
    return "Employs PCA to reduce the dimensionality of EEG/MEG data."; // TODO(kaehler): Comments
}

void WMPCA::connectors()
{
    m_input = boost::shared_ptr< LaBP::WLModuleInputDataRingBuffer< LaBP::WDataSetEMM > >(
                    new LaBP::WLModuleInputDataRingBuffer< LaBP::WDataSetEMM >( 32, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM > >(
                    new LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMPCA::properties()
{
    LaBP::WModuleEMMView::properties();

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
    std::vector< LaBP::WEModalityType::Enum > mEnums = LaBP::WEModalityType::values();
    for( std::vector< LaBP::WEModalityType::Enum >::iterator it = mEnums.begin(); it != mEnums.end(); ++it )
    {
        m_processModality->addItem(
                        boost::shared_ptr< WItemSelectionItemTyped< LaBP::WEModalityType::Enum > >(
                                        new WItemSelectionItemTyped< LaBP::WEModalityType::Enum >( *it,
                                                        LaBP::WEModalityType::name( *it ),
                                                        LaBP::WEModalityType::name( *it ) ) ) );
    }

    // getting the SelectorProperty from the list an add it to the properties
    m_processModalitySelection = m_propGrpPCAComputation->addProperty( "Process Modality",
                    "What kind of filter do you want to use", m_processModality->getSelectorFirst(),
                    boost::bind( &WMPCA::callbackProcessModalityChanged, this ) );

    // Be sure it is at least one selected, but not more than one
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_processModalitySelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_processModalitySelection );
}

void WMPCA::initModule()
{
    infoLog() << "Initializing module ...";
    waitRestored();

    initView( LaBP::WLEMDDrawable2D::WEGraphType::DYNAMIC ); // TODO(pieloth): No 3D needed!
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

    LaBP::WDataSetEMM::SPtr emmIn;

    ready(); // signal ready state

    initModule();

    debugLog() << "Entering main loop";
    while( !m_shutdownFlag() )
    {
        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() )
        {
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        emmIn.reset();
        if( !m_input->isEmpty() )
        {
            emmIn = m_input->getData();
        }
        const bool dataValid = ( emmIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            debugLog() << "received data";
            LaBP::WLTimeProfiler::SPtr profiler = emmIn->createAndAddProfiler( getName(), "process" );
            profiler->start();

            //m_pca->setParams( m_finalDimensions->get(), m_reverse->get() );
            LaBP::WDataSetEMM::SPtr emmOut( new LaBP::WDataSetEMM( *emmIn ) );

//            for( size_t mod = 0; mod < m_emm->getModalityCount(); ++mod )
//            {
            LaBP::WEModalityType::Enum mod = m_processModalitySelection->get().at( 0 )->getAs<
                            WItemSelectionItemTyped< LaBP::WEModalityType::Enum > >()->getValue();
            boost::shared_ptr< LaBP::WDataSetEMMEMD > emdIn = emmIn->getModality( mod );
            boost::shared_ptr< LaBP::WDataSetEMMEMD > emdOut = m_pca->processData( emdIn );

            // emm object can create in the outer module
            emmOut->addModality( emdOut );
//            }

            m_output->updateData( emmOut );
            updateView( emmOut );

            profiler->stopAndLog();
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
