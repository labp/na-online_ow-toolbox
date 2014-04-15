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

#include <vector>

#include <core/graphicsEngine/WGEZoomTrackballManipulator.h>
#include <core/kernel/WKernel.h>
#include <core/ui/WUI.h>
#include <core/ui/WUIWidgetFactory.h>
#include <core/ui/WUIViewWidget.h>

#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "WEEGSkinAlignment.h"
#include "WMAlignment.h"
#include "WMAlignment.xpm"

using namespace LaBP;

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMAlignment )

WMAlignment::WMAlignment()
{
    m_transformation.setZero();
}

WMAlignment::~WMAlignment()
{
}

const std::string WMAlignment::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Alignment";
}

const std::string WMAlignment::getDescription() const
{
    return "A semi-automatic coordinate transformation between the EEG sensor positions and the head model.";
}

WModule::SPtr WMAlignment::factory() const
{
    return WModule::SPtr( new WMAlignment() );
}

const char** WMAlignment::getXPMIcon() const
{
    return module_xpm;
}

void WMAlignment::connectors()
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

void WMAlignment::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );
    m_trgReset = m_properties->addProperty( "Reset:", "Reset", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_propEstGroup = m_properties->addPropertyGroup( "Transformation Estimation",
                    "Contains properties an initial transformation estimation.", false );

    m_propEstLPA = m_propEstGroup->addProperty( "LPA (AC-PC):", "Left pre-auricular in AC-PC coordinate system.", LAP, false );
    m_propEstNasion = m_propEstGroup->addProperty( "Nasion (AC-PC):", "Nasion in AC-PC coordinate system.", NASION, false );
    m_propEstRPA = m_propEstGroup->addProperty( "RPA (AC-PC):", "Right pre-auricular in AC-PC coordinate system.", RAP, false );

    m_propIcpGroup = m_properties->addPropertyGroup( "ICP properties", "Contains properties for ICP.", false );
    m_propIcpIterations = m_propIcpGroup->addProperty( "Iterations:", "Maximum iterations for ICP algorithm.",
                    ICP_DEFAULT_ITERATIONS, false );
    m_propIcpConverged = m_propIcpGroup->addProperty( "Converged:", "Indicates if ICP has converged.", false, false );
    m_propIcpConverged->setPurpose( PV_PURPOSE_INFORMATION );
    m_propIcpScore = m_propIcpGroup->addProperty( "Score:", "Fitness score of converged ICP.", WAlignment::NOT_CONVERGED, false );
    m_propIcpScore->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMAlignment::viewInit()
{
    WUIWidgetFactory::SPtr factory = WKernel::getRunningKernel()->getUI()->getWidgetFactory();
    m_widget = factory->createViewWidget( getName(), WGECamera::ORTHOGRAPHIC, m_shutdownFlag.getValueChangeCondition() );
    m_drawable = WLEMDDrawable3DEEGBEM::SPtr( new WLEMDDrawable3DEEGBEM( m_widget ) );
}

void WMAlignment::viewUpdate( WLEMMeasurement::SPtr emm )
{
    if( m_widget->isClosed() || !m_widget->isVisible() )
    {
        return;
    }
    m_drawable->draw( emm );
}

void WMAlignment::viewReset()
{
    m_drawable.reset( new WLEMDDrawable3DEEGBEM( m_widget ) );
}

void WMAlignment::moduleInit()
{
    infoLog() << "Initializing module ...";

    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // Wake up when input data changed
    m_moduleState.add( m_propCondition ); // Wake up when property data changed

    ready(); // signal ready state
    waitRestored();

    viewInit();

    infoLog() << "Initializing module finished!";

    infoLog() << "Restoring module ...";

    m_propIcpConverged->set( false, false );
    m_propIcpScore->set( WAlignment::NOT_CONVERGED, false );

    infoLog() << "Restoring module finished!";
}

void WMAlignment::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
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

        if( m_trgReset->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::RESET ) );
            processReset( cmd );
        }

        cmdIn.reset();
        if( !m_input->isEmpty() )
        {
            cmdIn = m_input->getData();
        }
        const bool dataValid = ( cmdIn );

        // ---------- INPUTDATAUPDATEEVENT ----------
        if( dataValid ) // If there was an update on the inputconnector
        {
            process( cmdIn );
        }
    }

    m_drawable.reset();
    m_widget->close();
    m_widget.reset();
}

bool WMAlignment::processCompute( WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WMAlignment", "processCompute" );

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );

    if( !m_transformation.isZero() )
    {
        emm->setFidToACPCTransformation( m_transformation );
        m_output->updateData( cmd );
        return true;
    }

    WEEGSkinAlignment align( m_propIcpIterations->get( false ) );
    align.setLpaSkin( m_propEstLPA->get( false ) );
    align.setNasionSkin( m_propEstNasion->get( false ) );
    align.setRpaSkin( m_propEstRPA->get( false ) );

    double score = align.align( &m_transformation, emm );
    if( score == WEEGSkinAlignment::NOT_CONVERGED )
    {
        m_output->updateData( cmd );
        m_propIcpConverged->set( false, false );
        return false;
    }
    m_propIcpConverged->set( true, false );
    m_propIcpScore->set( score, false );
    emm->setFidToACPCTransformation( m_transformation );
    viewUpdate( emm );

    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processInit( WLEMMCommand::SPtr cmd )
{
    if( cmd->hasEmm() )
    {
        WLTimeProfiler tp( "WMAlignment", "processInit" );

        const WLEMMeasurement::SPtr emm = cmd->getEmm();

        if( !m_transformation.isZero() )
        {
            emm->setFidToACPCTransformation( m_transformation );
            m_output->updateData( cmd );
            return true;
        }

        WEEGSkinAlignment align( m_propIcpIterations->get( false ) );
        align.setLpaSkin( m_propEstLPA->get( false ) );
        align.setNasionSkin( m_propEstNasion->get( false ) );
        align.setRpaSkin( m_propEstRPA->get( false ) );

        double score = align.align( &m_transformation, emm );
        if( score == WEEGSkinAlignment::NOT_CONVERGED )
        {
            m_output->updateData( cmd );
            m_propIcpConverged->set( false, false );
            m_output->updateData( cmd );
            return false;
        }
        m_propIcpConverged->set( true, false );
        m_propIcpScore->set( score, false );
        emm->setFidToACPCTransformation( m_transformation );
        viewUpdate( emm );
        m_output->updateData( cmd );
        return true;
    }
    else
    {
        m_output->updateData( cmd );
        return true;
    }
}

bool WMAlignment::processReset( WLEMMCommand::SPtr cmd )
{
    m_input->clear();
    handleTrgReset();
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processTime( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

bool WMAlignment::processMisc( WLEMMCommand::SPtr cmd )
{
    m_output->updateData( cmd );
    return true;
}

void WMAlignment::handleTrgReset()
{
    debugLog() << "handleTrgReset() called";

    viewReset();
    m_transformation.setZero();
    m_propEstLPA->set( LAP, false );
    m_propEstNasion->set( NASION, false );
    m_propEstRPA->set( RAP, false );
    m_propIcpConverged->set( false, false );
    m_propIcpScore->set( WAlignment::NOT_CONVERGED, false );
    m_propIcpIterations->set( ICP_DEFAULT_ITERATIONS, false );
    m_trgReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}
