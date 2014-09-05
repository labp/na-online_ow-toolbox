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

#include <typeinfo>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#include <osg/Array>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/gui/drawable/WLEMDDrawable3DSource.h"
#include "core/module/WLConstantsModule.h"
#include "core/util/profiler/WLTimeProfiler.h"
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

    /* init property container */
    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );
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

    // init the ROI-selector after the viewInit()-call
    WLEMMSurface::SPtr data;

    if( m_drawable3D->getAs< WLEMDDrawable3DSource >().get() )
    {
        m_drawable3D->getAs< WLEMDDrawable3DSource >()->setROISelector(
                        WLROISelectorSource::SPtr( new WLROISelectorSource( data, m_drawable3D ) ) );
        m_drawable3D->getAs< WLEMDDrawable3DSource >()->getROISelector()->getDirtyCondition()->subscribeSignal(
                        boost::function0< void >( boost::bind( &WMTemplateRoi::roiChanged, this ) ) );
    }

    infoLog() << "Initializing module finished!";
}

void WMTemplateRoi::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr emmIn;

    debugLog() << "Entering main loop";

    while( !m_shutdownFlag() )
    {
        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

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
    WLTimeProfiler tp( "WMTemplateRoi", "processCompute" );

    startUp( emm );

    debugLog() << "Modality Count: " << emm->getModalityCount();

    m_Emm = emm;

    // show process visualization
    boost::shared_ptr< WProgress > processComp = boost::shared_ptr< WProgress >( new WProgress( "Do the process." ) );
    m_progress->addSubProgress( processComp );

    // ---------- PROCESSING ----------
    m_drawable3D->getAs< WLEMDDrawable3DSource >()->getROISelector()->setData(
                    emm->getSubject()->getSurface( WLEMMSurface::Hemisphere::BOTH ) );
    viewUpdate( m_Emm );

    // ---------- OUTPUT ----------
    updateOutput();

    processComp->finish(); // finish the process visualization

    return true;
}

bool WMTemplateRoi::processInit( WLEMMCommand::SPtr labp )
{
    WProgress::SPtr progress( new WProgress( "Init view" ) );
    m_progress->addSubProgress( progress );

    progress->finish();
    m_progress->removeSubProgress( progress );

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

void WMTemplateRoi::updateOutput()
{
    updateOutput( m_Emm );
}

void WMTemplateRoi::updateOutput( WLEMMeasurement::SPtr emm )
{
    WLEMMCommand::SPtr cmd( new WLEMMCommand( WLEMMCommand::Command::COMPUTE ) );

    cmd->setEmm( emm );
    m_output->updateData( cmd ); // update the output-connector after processing
}

void WMTemplateRoi::roiChanged()
{
    boost::shared_ptr< const std::list< size_t > > filter =
                    m_drawable3D->getAs< WLEMDDrawable3DSource >()->getROISelector()->getFilter();

    if( filter->size() == 0 )
    {
        return;
    }

    if( !m_Emm->hasModality( WLEModality::SOURCE ) )
    {
        return;
    }

    WLEMData::DataT& data = m_Emm->getModality( WLEModality::SOURCE )->getData();
    data.setZero();

    //debugLog() << "Filter size: " << filter->size();
    //debugLog() << "Data rows: " << data.rows() << " cols: " << data.cols();

    for( std::list< size_t >::const_iterator it = filter->begin(); it != filter->end(); ++it )
    {
        long int i = *it;

        for( long int j = 0; j < data.cols(); ++j )
        {
            data( i, j ) = 1;
        }
    }

    updateOutput();
}

void WMTemplateRoi::startUp( WLEMMeasurement::SPtr emm )
{
    WLEMDSource::SPtr emdSrc( new WLEMDSource() );
    WLEMDSource::DataSPtr dataSrc( new WLEMDSource::DataT( 244662, 100 ) );
    dataSrc->setZero();
    emdSrc->setData( dataSrc );
    emm->addModality( emdSrc );

    WLEMDEEG::DataSPtr dataEEG( new WLEMDEEG::DataT( 60, 100 ) );
    dataEEG->setOnes();
    if( emm->hasModality( WLEModality::EEG ) )
    {
        emm->getModality( WLEModality::EEG )->setData( dataEEG );
    }
    else
    {
        WLEMDEEG::SPtr emdEEG( new WLEMDEEG() );
        emdEEG->setData( dataEEG );
        emm->addModality( emdEEG );
    }

    for( size_t i = 0; i < emm->getModalityCount(); ++i )
    {
        WLEMData::DataT& data = emm->getModality( i )->getData();
        debugLog() << WLEModality::name( emm->getModality( i )->getModalityType() ) << ": Rows: " << data.rows() << " Cols: "
                        << data.cols();
    }

    debugLog() << "Faces: " << emm->getSubject()->getSurface( WLEMMSurface::Hemisphere::BOTH )->getFaces()->size();
    debugLog() << "Vertices: " << emm->getSubject()->getSurface( WLEMMSurface::Hemisphere::BOTH )->getVertex()->size();
}
