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

#include <core/common/WLogger.h>

#include "core/io/WLReaderIsotrak.h"
#include "core/util/WLGeometry.h"
#include "WFTChunkNeuromagIsotrak.h"

const std::string WFTChunkNeuromagIsotrak::CLASS = "WFTChunkNeuromagIsotrak";

const int WFTChunkNeuromagIsotrak::EEG_FACES_FACTOR = -5;

WFTChunkNeuromagIsotrak::WFTChunkNeuromagIsotrak( const char* data, const size_t size ) :
                WFTAChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK, size )
{
    m_digPoints.reset( new WLList< WLDigPoint > );
    m_eegChPos.reset( new WLArrayList< WPosition > );
    m_eegFaces.reset( new WLArrayList< WVector3i > );

    processData( data, size );
}

WLList< WLDigPoint >::SPtr WFTChunkNeuromagIsotrak::getDigPoints() const
{
    return m_digPoints;
}

WLList< WLDigPoint >::SPtr WFTChunkNeuromagIsotrak::getDigPoints( WLEPointType::Enum type ) const
{
    WLList< WLDigPoint >::SPtr digForKind( new WLList< WLDigPoint >() );
    WLList< WLDigPoint >::const_iterator cit;
    for( cit = m_digPoints->begin(); cit != m_digPoints->end(); ++cit )
    {
        if( cit->getKind() == type )
        {
            digForKind->push_back( *cit );
        }
    }
    return digForKind;
}

WLArrayList< WPosition >::SPtr WFTChunkNeuromagIsotrak::getEEGChanPos() const
{
    return m_eegChPos;
}

WLArrayList< WVector3i >::SPtr WFTChunkNeuromagIsotrak::getEEGFaces() const
{
    return m_eegFaces;
}

bool WFTChunkNeuromagIsotrak::process( const char* data, size_t size )
{
    wlog::debug( CLASS ) << "process() called.";

    WLReaderIsotrak::SPtr reader( new WLReaderIsotrak( data, size ) );

    if( reader->read( m_digPoints ) != WLReader::ReturnCode::SUCCESS )
    {
        wlog::error( CLASS ) << "Neuromag header file was not read.";
        return false;
    }

    wlog::debug( CLASS ) << "DigPoints: " << m_digPoints->size();

    // generate EEG channel positions.
    if( createEEGPositions( getDigPoints( WLEPointType::EEG_ECG ) ) )
    {
        m_eegFaces->clear();
        WLGeometry::computeTriangulation( m_eegFaces.get(), *m_eegChPos, EEG_FACES_FACTOR );
    }

    return true;
}

WLSmartStorage::ConstSPtr WFTChunkNeuromagIsotrak::serialize() const
{
    WLSmartStorage::SPtr store( new WLSmartStorage );

    return store;
}

bool WFTChunkNeuromagIsotrak::createEEGPositions( WLList< WLDigPoint >::ConstSPtr digPoints )
{
    m_eegChPos->reserve( digPoints->size() );

    bool isFirst = true;
    WLList< WLDigPoint >::const_iterator it;
    for( it = digPoints->begin(); it != digPoints->end(); ++it )
    {
        if( it->getKind() != WLEPointType::EEG_ECG )
        {
            continue;
        }

        if( isFirst )
        {
            isFirst = false;
            continue;
        }

        m_eegChPos->push_back( it->getPoint() );
    }

    return !digPoints->empty() && !m_eegChPos->empty();
}
