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

#include <fstream>
#include <string>
#include <sstream>

#include <boost/shared_ptr.hpp>

#include <core/common/WItemSelectionItemTyped.h>
#include <core/common/WPathHelper.h>
#include <core/common/WPropertyHelper.h>
#include <core/kernel/WModule.h>

// Input & output connectors
// TODO use OW classes
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

// Input & output data
#include "core/data/WLDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/dataHandler/WDataSetEMMSource.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "core/util/WLTimeProfiler.h"

#include "WMEmdWriter.h"
#include "WMEmdWriter.xpm"

// This line is needed by the module loader to actually find your module.
W_LOADABLE_MODULE( WMEmdWriter )

WMEmdWriter::WMEmdWriter()
{
}

WMEmdWriter::~WMEmdWriter()
{

}

boost::shared_ptr< WModule > WMEmdWriter::factory() const
{
    return boost::shared_ptr< WModule >( new WMEmdWriter() );
}

const char** WMEmdWriter::getXPMIcon() const
{
    return module_xpm;
}

const std::string WMEmdWriter::getName() const
{
    return "EMD Writer";
}

const std::string WMEmdWriter::getDescription() const
{
    return "Writes EMD data to a file. Module supports LaBP data types only!";
}

void WMEmdWriter::connectors()
{
    m_input = boost::shared_ptr< LaBP::WLModuleInputDataRingBuffer< LaBP::WLDataSetEMM > >(
                    new LaBP::WLModuleInputDataRingBuffer< LaBP::WLDataSetEMM >( 8, shared_from_this(), "in",
                                    "Expects a EMM-DataSet for filtering." ) );
    addConnector( m_input );

    m_output = boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WLDataSetEMM > >(
                    new LaBP::WLModuleOutputDataCollectionable< LaBP::WLDataSetEMM >( shared_from_this(), "out",
                                    "Provides a filtered EMM-DataSet" ) );
    addConnector( m_output );
}

void WMEmdWriter::properties()
{
    LaBP::WLModuleDrawable::properties();

    m_propCondition = boost::shared_ptr< WCondition >( new WCondition() );

    m_propGrpModule = m_properties->addPropertyGroup( "Profile Logger", "Profile Logger", false );

    // Record stuff
    m_packetsAll = m_propGrpModule->addProperty( "Store all packets:", "Store all packets.", true, m_propCondition );
    m_packetsNext = m_propGrpModule->addProperty( "Store next packet only:", "Store next packet only.", false, m_propCondition );

    m_allEmd = m_propGrpModule->addProperty( "All modalities:", "Save all modalities", true );

    m_processModality = WItemSelection::SPtr( new WItemSelection() );
    std::vector< LaBP::WEModalityType::Enum > mEnums = LaBP::WEModalityType::values();
    for( std::vector< LaBP::WEModalityType::Enum >::iterator it = mEnums.begin(); it != mEnums.end(); ++it )
    {
        m_processModality->addItem(
                        WItemSelectionItemTyped< LaBP::WEModalityType::Enum >::SPtr(
                                        new WItemSelectionItemTyped< LaBP::WEModalityType::Enum >( *it,
                                                        LaBP::WEModalityType::name( *it ),
                                                        LaBP::WEModalityType::name( *it ) ) ) );
    }

    // getting the SelectorProperty from the list an add it to the properties
    m_processModalitySelection = m_propGrpModule->addProperty( "Modality to save", "Modality to save ...",
                    m_processModality->getSelectorFirst() ); // TODO

    // Be sure it is at least one selected, but not more than one
    WPropertyHelper::PC_SELECTONLYONE::addTo( m_processModalitySelection );
    WPropertyHelper::PC_NOTEMPTY::addTo( m_processModalitySelection );

    // File stuff
    const std::string folder( "/tmp/" );
    m_folder = m_propGrpModule->addProperty( "Folder:", "Folder ...", folder ); // TODO
    const std::string prefix( "labp_" );
    m_fPrefix = m_propGrpModule->addProperty( "File prefix:", "File prefix ...", prefix ); // TODO

    const std::string suffix( ".dat" );
    m_fSuffix = m_propGrpModule->addProperty( "File suffix:", "File suffix ...", suffix ); // TODO
}

void WMEmdWriter::initModule()
{
    infoLog() << "Initializing module ...";
    waitRestored();
    initView( LaBP::WLEMDDrawable2D::WEGraphType::DYNAMIC );
    infoLog() << "Initializing module finished!";
}

void WMEmdWriter::moduleMain()
{
    // init moduleState for using Events in mainLoop
    m_moduleState.setResetable( true, true ); // resetable, autoreset
    m_moduleState.add( m_input->getDataChangedCondition() ); // when inputdata changed
    m_moduleState.add( m_propCondition ); // when properties changed

    LaBP::WLDataSetEMM::SPtr emmIn;

    ready(); // signal ready state

    initModule();

    debugLog() << "Entering main loop";

    size_t count = 0;
    LaBP::WEModalityType::Enum emdType;
    bool allEmd;
    std::string folder;
    std::string prefix;
    std::string suffix;
    std::string fname;
    LaBP::WDataSetEMMEMD::ConstSPtr emd;

    while( !m_shutdownFlag() )
    {
        debugLog() << "Waiting for Events";
        if( m_input->isEmpty() ) // continue processing if data is available
        {
            m_moduleState.wait(); // wait for events like inputdata or properties changed
        }

        // ---------- SHUTDOWNEVENT ----------
        if( m_shutdownFlag() )
        {
            break; // break mainLoop on shutdown
        }

        if( m_packetsAll->changed( true ) )
        {
            if( m_packetsAll->get() )
            {
                m_packetsNext->set( false );
            }
        }
        if( m_packetsNext->changed( true ) )
        {
            if( m_packetsNext->get() )
            {
                m_packetsAll->set( false );
            }
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
            // TODO take every x. packet

            // The data is valid and we received an update. The data is not NULL but may be the same as in previous loops.
            debugLog() << "received data";

            if( m_packetsAll->get() || m_packetsNext->get() )
            {
                allEmd = m_allEmd->get();
                folder = m_folder->get();
                prefix = m_fPrefix->get();
                suffix = m_fSuffix->get();
                if( allEmd )
                {
                    for( size_t i = 0; i < emmIn->getModalityCount(); ++i )
                    {
                        emd = emmIn->getModality( i );
                        emdType = emd->getModalityType();
                        fname = getFileName( folder, prefix, suffix, emdType, emd->getNrChans(), emd->getSamplesPerChan(),
                                        count );
                        infoLog() << "file name: " << fname;
                        write( fname, emd );
                    }
                }
                else
                {
                    emdType = m_processModalitySelection->get().at( 0 )->getAs<
                                    WItemSelectionItemTyped< LaBP::WEModalityType::Enum > >()->getValue();
                    if( emmIn->hasModality( emdType ) )
                    {
                        emd = emmIn->getModality( emdType );
                        fname = getFileName( folder, prefix, suffix, emdType, emd->getNrChans(), emd->getSamplesPerChan(),
                                        count );
                        infoLog() << "file name: " << fname;
                        write( fname, emd );
                    }
                    else
                    {
                        warnLog() << "Selected modality is not available!";
                    }
                }
                m_packetsNext->set( false );
            }

            updateView( emmIn );
            ++count;
            m_output->updateData( emmIn );
        }
    }
}

bool WMEmdWriter::write( std::string fname, LaBP::WDataSetEMMEMD::ConstSPtr emd )
{
    std::ofstream fstream;
    fstream.open( fname.c_str(), std::ofstream::binary );

    if( emd->getModalityType() == LaBP::WEModalityType::SOURCE )
    {
        LaBP::MatrixT& data = emd->getAs< const LaBP::WDataSetEMMSource >()->getMatrix();
        const LaBP::MatrixT::Index channels = data.rows();
        const LaBP::MatrixT::Index samples = data.cols();
        LaBP::MatrixT::Scalar value;

        for( LaBP::MatrixT::Index i = 0; i < channels; ++i )
        {
            for( LaBP::MatrixT::Index j = 0; j < samples; ++j )
            {
                value = data( i, j );
                fstream.write( reinterpret_cast< char* >( &value ), sizeof( value ) );
            }
        }
    }
    else
    {
        LaBP::WDataSetEMMEMD::DataT& data = emd->getData();
        const size_t channels = emd->getNrChans();
        const size_t samples = emd->getSamplesPerChan();

        for( size_t i = 0; i < channels; ++i )
        {
            for( size_t j = 0; j < samples; ++j )
            {
                fstream.write( reinterpret_cast< char* >( &data[i][j] ), sizeof( data[i][j] ) );
            }
        }
    }

    fstream.close();
    return true;
}

std::string WMEmdWriter::getFileName( std::string folder, std::string prefix, std::string suffix,
                LaBP::WEModalityType::Enum emdType, size_t channels, size_t samples, size_t count )
{
    std::string fname = folder;
    fname.append( prefix );
    fname.append( LaBP::WEModalityType::name( emdType ) );
    std::stringstream ss;
    ss << "_" << channels << "x" << samples << "_" << count;
    fname.append( ss.str() );
    fname.append( suffix );
    return fname;
}
