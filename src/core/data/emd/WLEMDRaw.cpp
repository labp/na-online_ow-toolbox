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

#include <core/common/exceptions/WOutOfBounds.h>

#include "WLEMDRaw.h"

WLEMDRaw::WLEMDRaw() :
                WLEMData()
{
}

WLEMDRaw::~WLEMDRaw()
{
}

WLEMData::SPtr WLEMDRaw::clone() const
{
    WLEMDRaw::SPtr emd( new WLEMDRaw( *this ) );
    return emd;
}

WLEModality::Enum WLEMDRaw::getModalityType() const
{
    return WLEModality::UNKNOWN;
}

WLEMDRaw::DataSPtr WLEMDRaw::getData( const ChanPicksT& picks, bool checkIndices ) const
{
    if( checkIndices && ( m_data->rows() < picks.size() ) )
    {
        throw WOutOfBounds();
    }
    if( checkIndices && ( picks.minCoeff() < 0 || m_data->rows() < picks.maxCoeff() ) )
    {
        throw WOutOfBounds();
    }

    WLEMDRaw::DataSPtr dataOut( new WLEMDRaw::DataT( picks.size(), getSamplesPerChan() ) );
    for( ChanPicksT::Index i = 0; i < picks.size(); ++i )
    {
        dataOut->row( i ) = m_data->row( picks[i] );
    }
    return dataOut;
}
