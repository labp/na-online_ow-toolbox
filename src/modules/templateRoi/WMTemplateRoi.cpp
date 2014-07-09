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

#include <osg/Array>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

#include "core/module/WLConstantsModule.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "WLModelController.h"
#include "WLPickingHandler.h"
#include "WMTemplateRoi.h"

#include "WMTemplateRoi.xpm"

W_LOADABLE_MODULE( WMTemplateRoi )

WMTemplateRoi::WMTemplateRoi()
{

}

WMTemplateRoi::~WMTemplateRoi()
{

}

const std::string WMTemplateRoi::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " Template ROI";
}

const std::string WMTemplateRoi::getDescription() const
{
    return "A template module for testing the ROI functionality.";
}

void WMTemplateRoi::connectors()
{
    WLModuleDrawable::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMTemplateRoi::properties()
{
    WLModuleDrawable::properties();
    WLModuleDrawable::setTimerangeInformationOnly( true );
    WLModuleDrawable::setViewModality( WLEModality::SOURCE );
    WLModuleDrawable::hideComputeModalitySelection( true );
    WLModuleDrawable::hideViewModalitySelection( true );
    WLModuleDrawable::hideLabelChanged( true );

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    m_propGrpBox = m_properties->addPropertyGroup( "Box", "Box Properties", false );
    m_width = m_propGrpBox->addProperty( "Width", "Width", 20.0, false );
    m_width->setMin( 5 );
    m_width->setMax( 200 );
    m_height = m_propGrpBox->addProperty( "Height", "Height", 20.0, false );
    m_height->setMin( 5 );
    m_height->setMax( 200 );
    m_depth = m_propGrpBox->addProperty( "Depth", "Depth", 20.0, false );
    m_depth->setMin( 5 );
    m_depth->setMax( 200 );
}

void WMTemplateRoi::moduleInit()
{
    infoLog() << "Initializing module " << getName();

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

    viewInit( WLEMDDrawable2D::WEGraphType::SINGLE );

    infoLog() << "Initializing module finished!";
}

void WMTemplateRoi::moduleMain()
{
    moduleInit();

    initOSG();

    drawSome();

    WLEMMCommand::SPtr emmIn;

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        if( m_width->changed( true ) || m_height->changed( true ) || m_depth->changed( true ) )
        {
            resizeBox();
        }

        //m_moduleState.wait();

        /*
         if( m_input->isEmpty() ) // continue processing if data is available
         {
         m_moduleState.wait(); // wait for events like input-data or properties changed
         }

         // receive data form the input-connector
         emmIn.reset();
         if( !m_input->isEmpty() )
         {
         emmIn = m_input->getData();
         }
         const bool dataValid = ( emmIn );

         // ---------- INPUTDATAUPDATEEVENT ----------
         if( dataValid ) // If there was an update on the input-connector
         {
         // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
         debugLog() << "received data";

         process( emmIn );

         debugLog() << "finished processing";
         }
         */
    }
}

boost::shared_ptr< WModule > WMTemplateRoi::factory() const
{
    return boost::shared_ptr< WModule >( new WMTemplateRoi() );
}

const char** WMTemplateRoi::getXPMIcon() const
{
    return module_xpm;
}

bool WMTemplateRoi::processCompute( WLEMMeasurement::SPtr emm )
{
    WLTimeProfiler tp( "WMEpochRejection", "processCompute" );

    // show process visualization
    boost::shared_ptr< WProgress > processComp = boost::shared_ptr< WProgress >( new WProgress( "Do the process." ) );
    m_progress->addSubProgress( processComp );

    // ---------- PROCESSING ----------
    viewUpdate( emm ); // update the GUI component

    // ---------- OUTPUT ----------
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );
    cmd->setEmm( emm );
    m_output->updateData( cmd ); // update the output-connector after processing

    processComp->finish(); // finish the process visualization

    return true;
}

bool WMTemplateRoi::processInit( WLEMMCommand::SPtr labp )
{
    m_output->updateData( labp );
    return true;
}

bool WMTemplateRoi::processReset( WLEMMCommand::SPtr labp )
{
    viewReset();

    m_input->clear();
    m_output->updateData( labp );

    return true;
}

void WMTemplateRoi::initOSG()
{
    //m_drawable3D->getWidget()->add
}

void WMTemplateRoi::drawSome()
{
    m_geode = new osg::Geode();
    osg::ref_ptr< osg::ShapeDrawable > drawable = new osg::ShapeDrawable( new osg::Box( osg::Vec3( 0.0f, 0.0f, 0.0f ), 20.0f ) );
    drawable->setColor( osg::Vec4( 0.0f, 1.0f, 0.0f, 0.75f ) );
    m_geode->addDrawable( drawable );

    osg::ref_ptr< osg::MatrixTransform > mt = new osg::MatrixTransform;
    mt->addChild( m_geode.get() );

    m_drawable3D->getWidget()->getScene()->addChild( mt.get() );
    osg::ref_ptr< WLModelController > controller = new WLModelController( mt.get() );
    m_drawable3D->getWidget()->addEventHandler( controller );

    osg::ref_ptr< WLPickingHandler > picker = new WLPickingHandler;
    m_drawable3D->getWidget()->getScene()->addChild( picker->getOrCreateSelectionBox() );

    // TODO(maschke): the picker does not work!!
    //m_drawable3D->getWidget()->addEventHandler( picker.get() );
    m_drawable3D->getWidget()->getViewer()->getView()->addEventHandler( picker.get() );
}

void WMTemplateRoi::resizeBox()
{
    //debugLog() << "resizeBox() called!";

    if( m_geode->getDrawableList().size() > 0 )
    {
        osg::Box *ptr_box = ( osg::Box* )m_geode->getDrawable( 0 );
        osg::ShapeDrawable *ptr_drawable = ( osg::ShapeDrawable* )m_geode->getDrawable( 0 );
        const osg::Vec4 color = ptr_drawable->getColor();

        osg::ref_ptr< osg::ShapeDrawable > drawable = new osg::ShapeDrawable(
                        new osg::Box( ptr_box->getCenter(), m_width->get(), m_height->get(), m_depth->get() ) );
        drawable->setColor( color );

        m_geode->removeDrawable( m_geode->getDrawable( 0 ) );
        m_geode->addDrawable( drawable );

        m_drawable3D->redraw();
    }
}
