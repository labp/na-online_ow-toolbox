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

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMAlignment )

static const int ICP_DEFAULT_ITERATIONS = 10;

// Defaults for intershift is05
//    static const WPosition LAP( -0.0754, -0.0131, -0.0520 );
//    static const WPosition NASION( -0.0012, 0.0836, -0.0526 );
//    static const WPosition RAP( 0.0706, -0.0140, -0.0613 );
// Defaults for hermann
static const WPosition LAP( -0.07286011, 0.018106384, -0.068811984 );
static const WPosition NASION( 0.002131995, 0.098106384, -0.019811981 );
static const WPosition RAP( 0.075132007, 0.017106384, -0.074811978 );

WMAlignment::WMAlignment()
{
    m_transformation = WLTransformation::instance();
}

WMAlignment::~WMAlignment()
{
}

const std::string WMAlignment::getName() const
{
    return WLConstantsModule::generateModuleName( "Alignment" );
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
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMAlignment::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );
    m_trgReset = m_properties->addProperty( "Reset:", "Reset", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    m_propEstGroup = m_properties->addPropertyGroup( "Transformation Estimation",
                    "Contains properties an initial transformation estimation.", false );

    m_propEstLPA = m_propEstGroup->addProperty( "LPA (AC-PC) [m]:", "Left pre-auricular in AC-PC coordinate system and meter.",
                    LAP, false );
    m_propEstNasion = m_propEstGroup->addProperty( "Nasion (AC-PC) [m]:", "Nasion in AC-PC coordinate system and meter.", NASION,
                    false );
    m_propEstRPA = m_propEstGroup->addProperty( "RPA (AC-PC) [m]:", "Right pre-auricular in AC-PC coordinate system and meter.",
                    RAP, false );

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
    WLTimeProfiler tp( "WMAlignment", __func__ );

    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );

    if( !m_transformation->empty() )
    {
        emm->setFidToACPCTransformation( m_transformation );
        m_output->updateData( cmd );
        return true;
    }

    WEEGSkinAlignment align( m_propIcpIterations->get( false ) );
    align.setLpaSkin( m_propEstLPA->get( false ) );
    align.setNasionSkin( m_propEstNasion->get( false ) );
    align.setRpaSkin( m_propEstRPA->get( false ) );

    double score = align.align( m_transformation.get(), emm );
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
        WLTimeProfiler tp( "WMAlignment", __func__ );

        const WLEMMeasurement::SPtr emm = cmd->getEmm();
        if( !m_transformation->empty() )
        {
            emm->setFidToACPCTransformation( m_transformation );
            m_output->updateData( cmd );
            return true;
        }

        WEEGSkinAlignment align( m_propIcpIterations->get( false ) );
        align.setLpaSkin( m_propEstLPA->get( false ) );
        align.setNasionSkin( m_propEstNasion->get( false ) );
        align.setRpaSkin( m_propEstRPA->get( false ) );

        double score = align.align( m_transformation.get(), emm );
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
    hdlTrgReset();
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

void WMAlignment::hdlTrgReset()
{
    debugLog() << __func__ << "() called";

    viewReset();
    m_transformation = WLTransformation::instance();
    m_propEstLPA->set( LAP, false );
    m_propEstNasion->set( NASION, false );
    m_propEstRPA->set( RAP, false );
    m_propIcpConverged->set( false, false );
    m_propIcpScore->set( WAlignment::NOT_CONVERGED, false );
    m_propIcpIterations->set( ICP_DEFAULT_ITERATIONS, false );
    m_trgReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}
