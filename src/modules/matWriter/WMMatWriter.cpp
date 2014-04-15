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

#include <set>

#include <core/common/WException.h>
#include <core/common/WItemSelection.h>
#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/emd/WLEMData.h"
#include "core/module/WLConstantsModule.h"

#include "WMMatWriter.xpm"
#include "WMMatWriter.h"

W_LOADABLE_MODULE( WMMatWriter )

const std::string WMMatWriter::NONE = "none";
const std::string WMMatWriter::SUCCESS_WRITE = "Data successfully written.";
const std::string WMMatWriter::ERROR_WRITE = "Could not write data!";
const std::string WMMatWriter::ERROR_OPEN = "Could not open file!";
const std::string WMMatWriter::SUCCESS_OPEN = "Data successfully opened.";

WMMatWriter::WMMatWriter()
{
}

WMMatWriter::~WMMatWriter()
{
}

const std::string WMMatWriter::getName() const
{
    return WLConstantsModule::NAME_PREFIX + " MAT-File Writer";
}

const std::string WMMatWriter::getDescription() const
{
    return "Writes EMD object into a MATLAB MAT-file.";
}

WModule::SPtr WMMatWriter::factory() const
{
    return WModule::SPtr( new WMMatWriter );
}

const char** WMMatWriter::getXPMIcon() const
{
    return module_xpm;
}

void WMMatWriter::connectors()
{
    WModule::connectors();
    m_input.reset(
                    new WLModuleInputDataRingBuffer< WLEMMCommand >( 8, shared_from_this(), "in",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_input );
}

void WMMatWriter::properties()
{
    WModule::properties();

    m_propCondition = WCondition::SPtr( new WCondition() );

    m_propMatFile = m_properties->addProperty( "MAT-File:", "MATLAB MAT-File to write.", WPathHelper::getHomePath(),
                    m_propCondition );
    m_propMatFile->changed( true );

    WItemSelection::SPtr modSelection( new WItemSelection() );
    const std::set< WLEModality::Enum > modalities = WLEModality::values();
    std::set< WLEModality::Enum >::const_iterator it;
    for( it = modalities.begin(); it != modalities.end(); ++it )
    {
        modSelection->addItem(
                        WItemSelectionItemTyped< WLEModality::Enum >::SPtr(
                                        new WItemSelectionItemTyped< WLEModality::Enum >( *it, WLEModality::name( *it ),
                                                        WLEModality::description( *it ) ) ) );
    }

    m_selModality = m_properties->addProperty( "Modality:", "Modality to write.", modSelection->getSelectorFirst(),
                    m_propCondition );
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_selModality );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_selModality );

    m_status = m_properties->addProperty( "Status:", "Status", NONE );
    m_status->setPurpose( PV_PURPOSE_INFORMATION );
}

void WMMatWriter::moduleInit()
{
    infoLog() << "Initializing module ...";
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    ready(); // signal ready state
    waitRestored();
}

void WMMatWriter::moduleMain()
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
        if( m_shutdownFlag() )
        {
            break;
        }

        if( m_propMatFile->changed( true ) )
        {
            if( handleMatFileChanged() )
            {
                m_status->set( SUCCESS_OPEN, true );
            }
            else
            {
                m_status->set( ERROR_OPEN, true );
            }
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
            if( cmdIn->getCommand() == WLEMMCommand::Command::COMPUTE && cmdIn->hasEmm() )
            {
                if( writeData( cmdIn->getEmm() ) )
                {
                    m_status->set( SUCCESS_WRITE, true );
                }
                else
                {
                    m_status->set( ERROR_WRITE, true );
                }
            }
        }
    }
}

bool WMMatWriter::handleMatFileChanged()
{
    if( m_writer.get() != NULL )
    {
        m_writer->close();
    }
    const std::string fname = m_propMatFile->get().string();

    try
    {
        m_writer.reset( new WLWriterMAT( fname ) );
        WLIOStatus::ioStatus_t state = m_writer->init();
        if( state == WLIOStatus::SUCCESS )
        {
            infoLog() << SUCCESS_OPEN;
            return true;
        }
        else
        {
            errorLog() << m_writer->getIOStatusDescription( state );
            return false;
        }
    }
    catch( const WException& e )
    {
        errorLog() << ERROR_OPEN << "\n" << e.what();
        return false;
    }
}

bool WMMatWriter::writeData( WLEMMeasurement::ConstSPtr emmIn )
{
    if( !m_writer )
    {
        errorLog() << "No open writer available!";
        return false;
    }

    WLEModality::Enum mod =
                    m_selModality->get().at( 0 )->getAs< WItemSelectionItemTyped< WLEModality::Enum > >()->getValue();
    if( !emmIn->hasModality( mod ) )
    {
        errorLog() << "Modality data not available!";
        return false;
    }
    WLEMData::ConstSPtr emd = emmIn->getModality( mod );
    const WLEMData::DataT& data = emd->getData();
    WLIOStatus::ioStatus_t state = m_writer->writeMatrix( data );
    if( state == WLIOStatus::SUCCESS )
    {
        infoLog() << SUCCESS_WRITE;
        return true;
    }
    else
    {
        errorLog() << m_writer->getIOStatusDescription( state );
        return false;
    }
}
