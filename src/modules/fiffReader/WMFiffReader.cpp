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

#include <set>
#include <string>

#include <boost/filesystem.hpp>

#include <core/common/WException.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/kernel/WDataModuleInputFile.h>
#include <core/kernel/WDataModuleInputFilterFile.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/module/WLConstantsModule.h"
#include "core/io/WLReaderFIFF.h"
#include "core/io/WLReaderHpiInfo.h"
#include "core/util/WLGeometry.h"

#include "WMFiffReader.h"
#include "WMFiffReader.xpm"

using std::string;

W_LOADABLE_MODULE( WMFiffReader )

std::string WMFiffReader::EFileStatus::name( EFileStatus::Enum val )
{
    switch( val )
    {
        case NO_FILE:
            return "No file";
        case LOADING_FILE:
            return "Loading file ...";
        case FILE_ERROR:
            return "Could not read file!";
        case SUCCESS:
            return "File successfully read!";
        default:
            return "Unknown status!";
    }
}

std::string WMFiffReader::EDataStatus::name( EDataStatus::Enum val )
{
    switch( val )
    {
        case NO_DATA:
            return "No additional data";
        case DATA_AVAILABLE:
            return "Additional data available";
        case LOADING_DATA:
            return "Loading data ...";
        case DATA_ERROR:
            return "Could not load data!";
        case SUCCESS:
            return "Data successfully read!";
        default:
            return "Unknown status!";
    }
}

WMFiffReader::WMFiffReader() :
                WDataModule()
{
    m_fileStatus = EFileStatus::NO_FILE;
    m_dataStatus = EDataStatus::NO_DATA;
    m_reloadFiff = false;
    // FIXME #443: Workaround - crash on saving an empty WDataModule
    WDataModuleInputFile::SPtr phantom( new WDataModuleInputFile( WLConstantsModule::NO_FILE_FILENAME ) );
    setInput( phantom );
}

WMFiffReader::~WMFiffReader()
{
}

const std::string WMFiffReader::getName() const
{
    return WLConstantsModule::generateModuleName( "FIFF Reader" );
}

const std::string WMFiffReader::getDescription() const
{
    return "Reads a FIFF file and related/additional data if available.";
}

WModule::SPtr WMFiffReader::factory() const
{
    return WModule::SPtr( new WMFiffReader );
}

const char** WMFiffReader::getXPMIcon() const
{
    return module_xpm;
}

std::vector< WDataModuleInputFilter::ConstSPtr > WMFiffReader::getInputFilter() const
{
    std::vector< WDataModuleInputFilter::ConstSPtr > filters;
    filters.push_back( WDataModuleInputFilter::ConstSPtr( new WDataModuleInputFilterFile( "fif", "FIFF files" ) ) );
    return filters;
}

void WMFiffReader::handleInputChange()
{
    WDataModuleInputFile::SPtr inputFile = getInputAs< WDataModuleInputFile >();
    if( inputFile )
    {
        m_moduleState.notify();
        m_reloadFiff = true;
        return;
    }
    else
    {
        m_output->reset();
        m_emm.reset();
        m_subject.reset();
        updateFileStatus( EFileStatus::NO_FILE );
        updateDataStatus( EDataStatus::NO_DATA );
        return;
    }
}

void WMFiffReader::connectors()
{
    WModule::connectors();

    m_output = WLModuleOutputDataCollectionable< WLEMMCommand >::instance( shared_from_this(),
                    WLConstantsModule::CONNECTOR_NAME_OUT, WLConstantsModule::CONNECTOR_DESCR_OUT );
    addConnector( m_output );
}

void WMFiffReader::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_trgSendEMM = m_properties->addProperty( "Send EMM:", "Send", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );

    // FIFF file //
    m_propFileStatus = m_properties->addProperty( "File status:", "FIFF file status.",
                    EFileStatus::name( EFileStatus::NO_FILE ) );
    m_propFileStatus->setPurpose( PV_PURPOSE_INFORMATION );

    // Additional data //
    m_propDataStatus = m_properties->addProperty( "Data status:", "Status of additional data.",
                    EDataStatus::name( EDataStatus::NO_DATA ) );
    m_propDataStatus->setPurpose( PV_PURPOSE_INFORMATION );

    m_trgLoadData = m_properties->addProperty( "Load data", "Load data", WPVBaseTypes::PV_TRIGGER_READY, m_propCondition );
    m_trgLoadData->setHidden( true );

    const std::string subject( "none" );
    m_propSubject = m_properties->addProperty( "Subject:", "Selected a subject.", subject );
    m_propSubject->setHidden( true );
    m_propSubject->setPurpose( PV_PURPOSE_INFORMATION );

    const std::string trial( "none" );
    m_propTrial = m_properties->addProperty( "Trial:", "Selected a trial.", trial );
    m_propTrial->setHidden( true );
    m_propTrial->setPurpose( PV_PURPOSE_INFORMATION );

    m_itmBemFiles = WItemSelection::SPtr( new WItemSelection() );

    m_itmSurfaces = WItemSelection::SPtr( new WItemSelection() );
}

void WMFiffReader::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();

}

void WMFiffReader::moduleMain()
{
    moduleInit();

    while( !m_shutdownFlag() )
    {
        m_moduleState.wait();
        if( m_shutdownFlag() )
        {
            break;
        }

        if( m_reloadFiff )
        {
            hdlFiffFileChanged();
            m_reloadFiff = false;
        }

        if( m_trgLoadData->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgLoad();
        }

        if( m_trgSendEMM->get( true ) == WPVBaseTypes::PV_TRIGGER_TRIGGERED )
        {
            hdlTrgSendEMM();
        }
    }
}

void WMFiffReader::hdlTrgSendEMM()
{
    debugLog() << __func__ << "() called!";

    if( !m_emm )
    {
        errorLog() << "No EMM object to send!";
        return;
    }
    if( m_subject.get() )
    {
        m_emm->setSubject( m_subject );
    }

    WLEMMCommand::SPtr cmd = WLEMMCommand::instance( WLEMMCommand::Command::COMPUTE );
    cmd->setEmm( m_emm );
    m_output->updateData( cmd );
    infoLog() << "Sent EMM object.";
    m_trgSendEMM->set( WPVBaseTypes::PV_TRIGGER_READY, true );
}

void WMFiffReader::hdlFiffFileChanged()
{
    WDataModuleInputFile::SPtr inputFile = getInputAs< WDataModuleInputFile >();
    if( !inputFile )
    {
        updateDataStatus( EDataStatus::NO_DATA );
        return;
    }
    const std::string fName = inputFile->getFilename().string();

    WProgress::SPtr progress( new WProgress( "Reading FIFF file" ) );
    m_progress->addSubProgress( progress );

    updateFileStatus( EFileStatus::LOADING_FILE );
    if( readFiffFile( fName ) )
    {
        updateFileStatus( EFileStatus::SUCCESS );
        updateDataStatus( EDataStatus::LOADING_DATA );
        if( retrieveAdditionalData( fName ) )
        {
            updateDataStatus( EDataStatus::DATA_AVAILABLE );
        }
        else
        {
            updateDataStatus( EDataStatus::NO_DATA );
        }
    }
    else
    {
        updateFileStatus( EFileStatus::FILE_ERROR );
    }

    progress->finish();
    m_progress->removeSubProgress( progress );
}

bool WMFiffReader::readFiffFile( const std::string& fName )
{
    infoLog() << "Reading FIFF file: " << fName;
    try
    {
        WLReaderFIFF readerFiff( fName );
        m_emm.reset( new WLEMMeasurement() );
        if( readerFiff.read( &m_emm ) == WLIOStatus::SUCCESS )
        {
            infoLog() << "Read FIFF file.";
            if( m_emm->hasModality( WLEModality::EEG ) )
            {
                WLEMDEEG::SPtr eeg = m_emm->getModality< WLEMDEEG >( WLEModality::EEG );
                if( eeg->getFaces()->empty() )
                {
                    warnLog() << "No faces found! Faces will be generated.";
                    WLGeometry::computeTriangulation( eeg->getFaces().get(), eeg->getChannelPositions3d()->data(), -5 );
                }
            }
        }
    }
    catch( const WException& e )
    {
        errorLog() << "Error reading FIFF file: " << e.what();
        return false;
    }

    try
    {
        WLReaderHpiInfo readerHpi( fName );
        if( readerHpi.read( m_emm->getHpiInfo().get() ) == WLIOStatus::SUCCESS )
        {
            infoLog() << "Read HPI info.";
        }
        else
        {
            warnLog() << "No HPI info read!";
        }
    }
    catch( const WException& e )
    {
        errorLog() << "Error reading HPI info: " << e.what();
        return false;
    }

    infoLog() << "Reading FIFF file finished:";
    infoLog() << *m_emm;
    return true;
}

bool WMFiffReader::retrieveAdditionalData( const std::string& fName )
{
    bool hasData = false;
    boost::filesystem::path fiffFile( fName );
    boost::filesystem::path expRoot = WLReaderExperiment::getExperimentRootFromFiff( fiffFile );
    const std::string subject = WLReaderExperiment::getSubjectFromFiff( fiffFile );
    const std::string trial = WLReaderExperiment::getTrialFromFiff( fiffFile );

    m_propSubject->set( subject );
    m_propSubject->setHidden( false );
    m_propTrial->set( trial );
    m_propTrial->setHidden( false );

    if( !boost::filesystem::is_directory( expRoot ) )
    {
        expRoot = expRoot.parent_path();
    }
    m_expReader.reset( new WLReaderExperiment( expRoot.string(), subject ) );

    std::set< std::string > bems = m_expReader->findBems();
    m_itmBemFiles->clear();
    m_properties->removeProperty( m_selBemFiles );
    if( !bems.empty() )
    {
        hasData = true;
        std::set< std::string >::iterator itBEMs;
        for( itBEMs = bems.begin(); itBEMs != bems.end(); ++itBEMs )
        {
            m_itmBemFiles->addItem(
                            WItemSelectionItemTyped< string >::SPtr(
                                            new WItemSelectionItemTyped< string >( *itBEMs, *itBEMs ) ) );
        }
        m_selBemFiles = m_properties->addProperty( "BEM Layers:", "Select BEM Layers to use.",
                        m_itmBemFiles->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_selBemFiles );
        m_trgLoadData->setHidden( false );
    }

    m_itmSurfaces->clear();
    m_properties->removeProperty( m_selSurfaces );
    std::set< std::string > surfaces = m_expReader->findSurfaceKinds();
    if( !surfaces.empty() )
    {
        hasData = true;
        std::set< std::string >::iterator itSurfs;
        for( itSurfs = surfaces.begin(); itSurfs != surfaces.end(); ++itSurfs )
        {
            m_itmSurfaces->addItem(
                            WItemSelectionItemTyped< string >::SPtr(
                                            new WItemSelectionItemTyped< string >( *itSurfs, *itSurfs ) ) );
        }
        m_selSurfaces = m_properties->addProperty( "Surfaces:", "Select surface to use.", m_itmSurfaces->getSelectorFirst() );

        // Be sure it is at least one selected, but not more than one
        WPropertyHelper::PC_SELECTONLYONE::addTo( m_selSurfaces );
        m_trgLoadData->setHidden( false );
    }

    return hasData;
}

void WMFiffReader::hdlTrgLoad()
{
    WProgress::SPtr progress( new WProgress( "Reading additional data" ) );
    m_progress->addSubProgress( progress );

    infoLog() << "Reading additional data ...";
    updateDataStatus( EDataStatus::LOADING_DATA );
    if( readData() )
    {
        updateDataStatus( EDataStatus::SUCCESS );
    }
    else
    {
        updateDataStatus( EDataStatus::DATA_ERROR );
    }
    infoLog() << "Finish reading additional data!";
    m_trgLoadData->set( WPVBaseTypes::PV_TRIGGER_READY, true );

    progress->finish();
    m_progress->removeSubProgress( progress );
}

bool WMFiffReader::readData()
{
    bool succes = false;
    m_subject.reset( new WLEMMSubject() );

    const string bemFile = m_selBemFiles->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    succes |= m_expReader->readBem( bemFile, m_subject );

    const string surfaceType = m_selSurfaces->get().at( 0 )->getAs< WItemSelectionItemTyped< string > >()->getValue();
    succes |= m_expReader->readSourceSpace( surfaceType, m_subject );

    const string trial = m_propTrial->get();
    succes |= m_expReader->readLeadFields( surfaceType, bemFile, trial, m_subject );

    return succes;
}
