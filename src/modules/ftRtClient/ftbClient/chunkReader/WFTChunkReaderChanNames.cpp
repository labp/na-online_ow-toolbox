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

#include <algorithm>  // transform
#include <list>
#include <string>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDECG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "modules/ftRtClient/ftb/WFtbChunk.h"

#include "WFTChunkReaderChanNames.h"

using namespace std;

const std::string WFTChunkReaderChanNames::CLASS = "WFTChunkReaderChanNames";

WFTChunkReaderChanNames::WFTChunkReaderChanNames()
{
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eeg", WLEModality::EEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "meg", WLEModality::MEG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "eog", WLEModality::EOG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "ecg", WLEModality::ECG ) );
    m_modalityLabels.insert( ChanNameLabelT::value_type( "misc", WLEModality::UNKNOWN ) );
}

WFTChunkReaderChanNames::~WFTChunkReaderChanNames()
{
}

wftb::chunk_type_t WFTChunkReaderChanNames::supportedChunkType() const
{
    return wftb::ChunkType::CHANNEL_NAMES;
}

bool WFTChunkReaderChanNames::read( WFTChunk::ConstSPtr chunk )
{
    wlog::debug( CLASS ) << __func__ << "() called.";

    if( chunk->getChunkType() != supportedChunkType() )
    {
        wlog::error( CLASS ) << "Chunk type not supported: " << wftb::ChunkType::name( chunk->getChunkType() );
        return false;
    }

    m_namesMap.clear();
    m_namesAll.clear();
    std::string str( ( const char* )chunk->getData(), chunk->getDataSize() );
    int chans = 0;

    split( &m_namesAll, str, '\0' );

    if( m_namesAll.empty() )
    {
        return false;
    }

    BOOST_FOREACH( std::string chanName, m_namesAll ){
    WLEModality::Enum modality = WLEModality::UNKNOWN;
    bool found = false;
    std::string channel( chanName );
    boost::algorithm::to_lower( channel );
    std::pair< std::string, WLEModality::Enum > label;

    BOOST_FOREACH( label, m_modalityLabels )
    {
        found = channel.find( label.first ) != std::string::npos;
        if( found )
        {
            modality = label.second;
            break;
        }
    }

    if( !found )
    {
        wlog::debug( CLASS ) << "Reject channel name: " << chanName;
        continue;
    }

    if( m_namesMap.count( modality ) == 0 )
    {
        m_namesMap.insert(
                        ChanNamesMapT::value_type( modality,
                                        WLArrayList< std::string >::SPtr( new WLArrayList< std::string >() ) ) );
    }

    m_namesMap.at( modality )->push_back( chanName );
    ++chans;
}

    wlog::debug( CLASS ) << "Channel names read. Number of assigned channels: " << chans;
    wlog::debug( CLASS ) << "Channel names in string vector: " << m_namesAll.size();
    std::pair< WLEModality::Enum, WLArrayList< std::string >::SPtr > list;
    BOOST_FOREACH( list, m_namesMap ){
    wlog::debug( CLASS ) << "Channel names for modality " << list.first << ": " << list.second->size();
}

    return m_namesMap.size() > 0;
}

bool WFTChunkReaderChanNames::apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw )
{
    if( emm->getModalityCount() == 0 && !m_namesAll.empty() )
    {
        return extractEmdsByNames( emm, raw );
    }

    if( emm->getModalityCount() > 0 && !m_namesMap.empty() )
    {
        return setChannelNames( emm );
    }

    return false;
}

bool WFTChunkReaderChanNames::setChannelNames( WLEMMeasurement::SPtr emm )
{
    bool rc = false;

    std::vector< WLEMData::SPtr > emds = emm->getModalityList();
    std::vector< WLEMData::SPtr >::iterator it;
    for( it = emds.begin(); it != emds.end(); ++it )
    {
        const WLEModality::Enum mod = ( *it )->getModalityType();
        if( m_namesMap.count( mod ) > 0 )
        {
            ( *it )->setChanNames( m_namesMap[mod] );
            rc |= true;
        }
    }
    return rc;
}

bool WFTChunkReaderChanNames::extractEmdsByNames( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw )
{
    if( static_cast< WLChanNrT >( m_namesAll.size() ) != raw->getNrChans() )
    {
        wlog::error( CLASS ) << __func__ << ": Channel count is different!";
        return false;
    }

    bool rc = false;
    WLEMData::SPtr emd;

    emd.reset( new WLEMDEEG() );
    if( extractEmdByName( "eeg", emd, raw ) )
    {
        emm->addModality( emd );
        rc |= true;
    }

    emd.reset( new WLEMDMEG() );
    if( extractEmdByName( "meg", emd, raw ) )
    {
        emm->addModality( emd );
        rc |= true;
    }

    emd.reset( new WLEMDEOG() );
    if( extractEmdByName( "eog", emd, raw ) )
    {
        emm->addModality( emd );
        rc |= true;
    }

    emd.reset( new WLEMDECG() );
    if( extractEmdByName( "ecg", emd, raw ) )
    {
        emm->addModality( emd );
        rc |= true;
    }

    return rc;
}

bool WFTChunkReaderChanNames::extractEmdByName( std::string idLower, WLEMData::SPtr emd, WLEMDRaw::ConstSPtr raw )
{
    std::transform( idLower.begin(), idLower.end(), idLower.begin(), ::tolower );
    std::string idUpper = idLower;
    std::transform( idUpper.begin(), idUpper.end(), idUpper.begin(), ::toupper );

    WLEMDRaw::ChanPicksT picks( m_namesAll.size() );
    WLEMDRaw::ChanPicksT::Index i = 0;
    WLEMDRaw::ChanPicksT::Index p = 0;
    std::list< std::string >::const_iterator it;
    for( it = m_namesAll.begin(); it != m_namesAll.end(); ++it )
    {
        if( it->find( idLower ) != std::string::npos )
        {
            picks( p++ ) = i;
            ++i;
            continue;
        }
        if( it->find( idUpper ) != std::string::npos )
        {
            picks( p++ ) = i;
            ++i;
            continue;
        }
        ++i;
    }

    if( p == 0 )
    {
        return false;
    }

    WLEMData::DataSPtr data = raw->getData( picks.segment( 0, p ) );
    emd->setData( data );
    emd->setSampFreq( raw->getSampFreq() );
    if( m_namesMap.count( emd->getModalityType() ) > 0 )
    {
        emd->setChanNames( m_namesMap[emd->getModalityType()] );
    }
    return true;
}

void WFTChunkReaderChanNames::split( std::list< std::string >* const result, const std::string& str, const char delim )
{
    result->clear();

    size_t end = -1;
    size_t start = 0;

    int counter = 0;

    BOOST_FOREACH( char ch, str ){
    ++end;

    std::string cmp_str( &ch, 1 );

    if( ch == delim )
    {
        ++counter;

        result->push_back( str.substr( start, end - start ) );
        start = end + 1;
    }
}
}
