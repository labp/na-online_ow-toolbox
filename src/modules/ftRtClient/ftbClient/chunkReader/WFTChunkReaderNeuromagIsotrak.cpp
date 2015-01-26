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

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMDEEG.h"
#include "core/io/WLReaderIsotrak.h"
#include "core/util/WLGeometry.h"

#include "modules/ftRtClient/ftb/WFtbChunk.h"
#include "WFTChunkReaderNeuromagIsotrak.h"

const std::string WFTChunkReaderNeuromagIsotrak::CLASS = "WFTChunkReaderNeuromagIsotrak";

const int WFTChunkReaderNeuromagIsotrak::EEG_FACES_FACTOR = -5;

WFTChunkReaderNeuromagIsotrak::WFTChunkReaderNeuromagIsotrak()
{
    m_digPoints.reset( new WLList< WLDigPoint > );
    m_eegChPos = WLPositions::instance();
    m_eegFaces.reset( new WLArrayList< WVector3i > );
}

WFTChunkReaderNeuromagIsotrak::~WFTChunkReaderNeuromagIsotrak()
{
}

wftb::chunk_type_t WFTChunkReaderNeuromagIsotrak::supportedChunkType() const
{
    return wftb::ChunkType::NEUROMAG_ISOTRAK;
}

bool WFTChunkReaderNeuromagIsotrak::read( WFTChunk::ConstSPtr chunk )
{
    wlog::debug( CLASS ) << __func__ << "() called.";

    if( chunk->getChunkType() != supportedChunkType() )
    {
        wlog::error( CLASS ) << "Chunk type not supported: " << wftb::ChunkType::name( chunk->getChunkType() );
        return false;
    }

    m_digPoints.reset( new WLList< WLDigPoint > );
    m_eegChPos = WLPositions::instance();
    // TODO(pieloth): #393 set unit and exponent.
    m_eegChPos->coordSystem( WLECoordSystem::HEAD );
    m_eegFaces.reset( new WLArrayList< WVector3i > );

    WLReaderIsotrak::SPtr reader( new WLReaderIsotrak( ( const char* )chunk->getData(), chunk->getDataSize() ) );
    if( reader->read( m_digPoints.get() ) != WLIOStatus::SUCCESS )
    {
        wlog::error( CLASS ) << "Neuromag header file was not read.";
        return false;
    }

    wlog::debug( CLASS ) << "DigPoints: " << m_digPoints->size();

    // generate EEG channel positions.
    if( createEEGPositions( getDigPoints( WLEPointType::EEG_ECG ) ) )
    {
        m_eegFaces->clear();
        WLGeometry::computeTriangulation( m_eegFaces.get(), m_eegChPos->data(), EEG_FACES_FACTOR );
    }

    return true;
}

bool WFTChunkReaderNeuromagIsotrak::apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw )
{
    bool rc = false;
    if( !getDigPoints()->empty() )
    {
        emm->setDigPoints( getDigPoints() );
        rc |= true;
    }

    if( !getEEGChanPos()->empty() && emm->hasModality( WLEModality::EEG ) )
    {
        WLEMDEEG::SPtr eeg = emm->getModality< WLEMDEEG >( WLEModality::EEG );
        eeg->setChannelPositions3d( getEEGChanPos() );
        if( !getEEGFaces()->empty() )
        {
            eeg->setFaces( getEEGFaces() );
        }
        rc |= true;
    }

    return rc;
}

WLList< WLDigPoint >::SPtr WFTChunkReaderNeuromagIsotrak::getDigPoints() const
{
    return m_digPoints;
}

WLList< WLDigPoint >::SPtr WFTChunkReaderNeuromagIsotrak::getDigPoints( WLEPointType::Enum type ) const
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

WLPositions::SPtr WFTChunkReaderNeuromagIsotrak::getEEGChanPos() const
{
    return m_eegChPos;
}

WLArrayList< WVector3i >::SPtr WFTChunkReaderNeuromagIsotrak::getEEGFaces() const
{
    return m_eegFaces;
}

bool WFTChunkReaderNeuromagIsotrak::createEEGPositions( WLList< WLDigPoint >::ConstSPtr digPoints )
{
    bool isFirst = true;
    WLList< WLDigPoint >::const_iterator it;
    WLPositions::IndexT nPos = 0;
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
        ++nPos;
    }

    m_eegChPos->resize( nPos );
    nPos = 0;
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
        WLPositions::PositionT tmp( it->getPoint().x(), it->getPoint().y(), it->getPoint().z() );
        m_eegChPos->data().col( nPos ) = tmp;
        ++nPos;
    }

    return !digPoints->empty() && !m_eegChPos->empty();
}
