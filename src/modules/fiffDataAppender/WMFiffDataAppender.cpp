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

#include <core/common/WPathHelper.h>

#include "core/container/WLList.h"
#include "core/io/WLReaderBem.h"
#include "core/io/WLReaderHpiInfo.h"
#include "core/io/WLReaderLeadfield.h"
#include "core/io/WLReaderSourceSpace.h"
#include "core/module/WLConstantsModule.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "WMFiffDataAppender.h"
#include "WMFiffDataAppender.xpm"

W_LOADABLE_MODULE( WMFiffDataAppender )

static const std::string DATA_NOT_LOADED = "No data loaded.";
static const std::string DATA_LOADING = "Loading data ...";
static const std::string DATA_LOADED = "Data successfully loaded.";
static const std::string DATA_ERROR = "Could not load data.";

WMFiffDataAppender::WMFiffDataAppender()
{
}

WMFiffDataAppender::~WMFiffDataAppender()
{
}

WModule::SPtr WMFiffDataAppender::factory() const
{
    return WModule::SPtr( new WMFiffDataAppender );
}

const std::string WMFiffDataAppender::getName() const
{
    return WLConstantsModule::generateModuleName( "FIFF Data Appender" );
}

const std::string WMFiffDataAppender::getDescription() const
{
    return "Appends an EMM block by additional data, e.g. source space.";
}

const char** WMFiffDataAppender::getXPMIcon() const
{
    return module_xpm;
}

void WMFiffDataAppender::connectors()
{
    WModule::connectors();

    m_input = WLModuleInputDataRingBuffer< WLEMMCommand >::instance( WLConstantsModule::BUFFER_SIZE, shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_IN, WLConstantsModule::CONNECTOR_DESCR_IN );
    addConnector( m_input );

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMFiffDataAppender::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_srcSpaceFile = m_properties->addProperty( "Source space file:", "Read a FIFF file containing the source space.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_srcSpaceFile->changed( true );

    m_bemFile = m_properties->addProperty( "BEM file:", "Read a FIFF file containing BEM layers.", WPathHelper::getHomePath(),
                    m_propCondition );
    m_bemFile->changed( true );

    m_lfEEGFile = m_properties->addProperty( "Leadfield EEG file:", "Read a FIFF file containing the leadfield for EEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfEEGFile->changed( true );

    m_lfMEGFile = m_properties->addProperty( "Leadfield MEG file:", "Read a FIFF file containing the leadfield for MEG.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_lfMEGFile->changed( true );

    m_hpiInfoFile = m_properties->addProperty( "HPI info file:", "Read HPI information from a FIFF file.",
                    WPathHelper::getHomePath(), m_propCondition );
    m_hpiInfoFile->changed( true );

    m_propStatus = m_properties->addProperty( "Status:", "Data status.", DATA_NOT_LOADED );
    m_propStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgReset = m_properties->addProperty( "Reset:", "Reset", WPVBaseTypes::PV_TRIGGER_READY,
                    boost::bind( &WMFiffDataAppender::cbReset, this ) );
}

void WMFiffDataAppender::moduleInit()
{

    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() );
    m_moduleState.add( m_propCondition ); // wake up when properties changed

    ready(); // signal ready state
    waitRestored();

    infoLog() << "Restoring module ...";

    if( m_srcSpaceFile->changed( true ) )
    {
        hdlSurfaceFileChanged( m_srcSpaceFile->get().string() );
    }
    if( m_bemFile->changed( true ) )
    {
        hdlBemFileChanged( m_bemFile->get().string() );
    }

    if( m_lfEEGFile->changed( true ) )
    {
        hdlLeadfieldFileChanged( &m_leadfieldEEG, m_lfEEGFile->get().string() );
    }
    if( m_lfMEGFile->changed( true ) )
    {
        hdlLeadfieldFileChanged( &m_leadfieldMEG, m_lfMEGFile->get().string() );
    }

    if( m_hpiInfoFile->changed( true ) )
    {
        hdlHpiInfoFileChanged( m_hpiInfoFile->get().string() );
    }

    infoLog() << "Restoring module finished!";
}

void WMFiffDataAppender::moduleMain()
{
    moduleInit();

    WLEMMCommand::SPtr cmdIn;
    while( !m_shutdownFlag() )
    {
        if( m_input->isEmpty() )
        {
            m_moduleState.wait();
        }

        if( m_srcSpaceFile->changed( true ) )
        {
            hdlSurfaceFileChanged( m_srcSpaceFile->get().string() );
        }
        if( m_bemFile->changed( true ) )
        {
            hdlBemFileChanged( m_bemFile->get().string() );
        }

        if( m_lfEEGFile->changed( true ) )
        {
            hdlLeadfieldFileChanged( &m_leadfieldEEG, m_lfEEGFile->get().string() );
        }
        if( m_lfMEGFile->changed( true ) )
        {
            hdlLeadfieldFileChanged( &m_leadfieldMEG, m_lfMEGFile->get().string() );
        }

        if( m_hpiInfoFile->changed( true ) )
        {
            hdlHpiInfoFileChanged( m_hpiInfoFile->get().string() );
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
            m_output->updateData( cmdIn );
        }
    }
}

bool WMFiffDataAppender::hdlLeadfieldFileChanged( WLMatrix::SPtr* const lf, std::string fName )
{
    debugLog() << __func__ << "()";

    WProgress::SPtr progress( new WProgress( "Reading Leadfield" ) );
    m_progress->addSubProgress( progress );
    m_propStatus->set( DATA_LOADING, true );

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
    {
        m_propStatus->set( DATA_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read leadfield!";
        m_propStatus->set( DATA_ERROR, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMFiffDataAppender::hdlSurfaceFileChanged( std::string fName )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Reading Surface" ) );
    m_progress->addSubProgress( progress );
    m_propStatus->set( DATA_LOADING, true );

    WLReaderSourceSpace::SPtr reader;
    try
    {
        reader.reset( new WLReaderSourceSpace( fName ) );
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }

    m_surface.reset( new WLEMMSurface() );
    if( reader->read( &m_surface ) == WLIOStatus::SUCCESS )
    {
        m_propStatus->set( DATA_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read source space!";
        m_propStatus->set( DATA_ERROR, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMFiffDataAppender::hdlBemFileChanged( std::string fName )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Reading BEM Layer" ) );
    m_progress->addSubProgress( progress );
    m_propStatus->set( DATA_LOADING, true );

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
    if( reader->read( m_bems.get() ) == WLIOStatus::SUCCESS )
    {
        infoLog() << "Loaded BEM layer: " << m_bems->size();
        m_propStatus->set( DATA_LOADED, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return true;
    }
    else
    {
        errorLog() << "Could not read BEM layers!";
        m_propStatus->set( DATA_ERROR, true );
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

bool WMFiffDataAppender::hdlHpiInfoFileChanged( std::string fName )
{
    debugLog() << __func__ << "() called!";

    WProgress::SPtr progress( new WProgress( "Reading HPI info" ) );
    m_progress->addSubProgress( progress );
    m_propStatus->set( DATA_LOADING, true );

    m_hpiInfo.reset();
    try
    {
        WLReaderHpiInfo reader( fName );
        WLEMMHpiInfo::SPtr hpiInfo( new WLEMMHpiInfo() );
        if( reader.read( hpiInfo.get() ) == WLIOStatus::SUCCESS )
        {
            infoLog() << "Read HPI info.";
            infoLog() << *hpiInfo;
            m_hpiInfo = hpiInfo;
            m_propStatus->set( DATA_LOADED, true );
            progress->finish();
            m_progress->removeSubProgress( progress );
            return true;
        }
        else
        {
            errorLog() << "Could not read HPI info!";
            m_propStatus->set( DATA_ERROR, true );
            progress->finish();
            m_progress->removeSubProgress( progress );
            return false;
        }
    }
    catch( const WDHNoSuchFile& e )
    {
        errorLog() << "File does not exist: " << fName;
        progress->finish();
        m_progress->removeSubProgress( progress );
        return false;
    }
}

void WMFiffDataAppender::cbReset()
{
    debugLog() << __func__ << "() called!";

    m_surface.reset();
    m_srcSpaceFile->set( WPathHelper::getHomePath().string(), true );
    m_srcSpaceFile->changed( true );

    if( m_bems )
    {
        m_bems->clear();
        m_bems.reset();
    }
    m_bemFile->set( WPathHelper::getHomePath().string(), true );
    m_bemFile->changed( true );

    m_leadfieldEEG.reset();
    m_leadfieldMEG.reset();
    m_lfEEGFile->set( WPathHelper::getHomePath().string(), true );
    m_lfEEGFile->changed( true );
    m_lfMEGFile->set( WPathHelper::getHomePath().string(), true );
    m_lfMEGFile->changed( true );

    m_hpiInfo.reset();

    m_propStatus->set( DATA_NOT_LOADED, true );
    m_trgReset->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMFiffDataAppender::process( WLEMMCommand::SPtr cmd )
{
    if( !cmd->hasEmm() )
    {
        debugLog() << __func__ << ": Command contains no EMM.";
        return;
    }

    WLEMMeasurement::SPtr emm = cmd->getEmm();
    WLEMMSubject::SPtr subject = emm->getSubject();

    if( m_bems )
    {
        subject->setBemBoundaries( m_bems );
    }
    if( m_leadfieldEEG )
    {
        subject->setLeadfield( WLEModality::EEG, m_leadfieldEEG );
    }
    if( m_leadfieldMEG )
    {
        subject->setLeadfield( WLEModality::MEG, m_leadfieldMEG );
    }
    if( m_surface )
    {
        subject->setSurface( m_surface );
    }
    if( m_hpiInfo )
    {
        emm->setHpiInfo( m_hpiInfo );
    }
}
