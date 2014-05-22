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

#include "WFTChunkNeuromagIsotrak.h"

const std::string WFTChunkNeuromagIsotrak::CLASS = "WFTChunkNeuromagIsotrak";

WFTChunkNeuromagIsotrak::WFTChunkNeuromagIsotrak( const char* data, const size_t size ) :
                WFTAChunk( WLEFTChunkType::FT_CHUNK_NEUROMAG_ISOTRAK, size )
{
    processData( data, size );
}

WLList< WLDigPoint >::SPtr WFTChunkNeuromagIsotrak::getData() const
{
    return m_digPoints;
}

bool WFTChunkNeuromagIsotrak::process( const char* data, size_t size )
{
    wlog::debug( CLASS ) << "process() called.";

    m_digPoints.reset( new WLList< WLDigPoint > );

    WLReaderIsotrak::SPtr reader( new WLReaderIsotrak( data, size ) );

    if( reader->read( m_digPoints ) != WLReader::ReturnCode::SUCCESS )
    {
        wlog::error( CLASS ) << "Neuromag header file was not read.";
        return false;
    }

    wlog::debug( CLASS ) << "DigPoints: " << m_digPoints->size();

    return true;
}

WLSmartStorage::ConstSPtr WFTChunkNeuromagIsotrak::serialize() const
{
    WLSmartStorage::SPtr store( new WLSmartStorage );

    // TODO(maschke): serialize isotrak into smart storage.

    return store;
}
