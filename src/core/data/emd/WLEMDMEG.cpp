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

#include <boost/shared_ptr.hpp>

#include <core/common/WAssert.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMEnumTypes.h"

#include "WLEMData.h"
#include "WLEMDMEG.h"

WLEMDMEG::WLEMDMEG() :
                WLEMData()
{
    m_chanPos3d = boost::shared_ptr< std::vector< WPosition > >( new std::vector< WPosition >() );
    m_faces = boost::shared_ptr< std::vector< WVector3i > >( new std::vector< WVector3i >() );

    m_eX.reset( new std::vector< WVector3f >() );
    m_eY.reset( new std::vector< WVector3f >() );
    m_eZ.reset( new std::vector< WVector3f >() );
}

WLEMDMEG::WLEMDMEG( const WLEMDMEG& meg ) :
                WLEMData( meg )
{
    m_chanPos3d = meg.m_chanPos3d;
    m_faces = meg.m_faces;
    m_eX = meg.m_eX;
    m_eY = meg.m_eY;
    m_eZ = meg.m_eZ;
}

WLEMDMEG::~WLEMDMEG()
{
}

WLEMData::SPtr WLEMDMEG::clone() const
{
    WLEMDMEG::SPtr meg( new WLEMDMEG( *this ) );
    return meg;
}

LaBP::WEModalityType::Enum WLEMDMEG::getModalityType() const
{
    return LaBP::WEModalityType::MEG;
}

boost::shared_ptr< std::vector< WPosition > > WLEMDMEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void WLEMDMEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    this->m_chanPos3d = chanPos3d;
}

std::vector< WVector3i >& WLEMDMEG::getFaces() const
{
    return *m_faces;
}

void WLEMDMEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

std::vector< WVector3f >& WLEMDMEG::getEx() const
{
    return *m_eX;
}

void WLEMDMEG::setEx( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eX = vec;
}

std::vector< WVector3f >& WLEMDMEG::getEy() const
{
    return *m_eY;
}

void WLEMDMEG::setEy( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eY = vec;
}

std::vector< WVector3f >& WLEMDMEG::getEz() const
{
    return *m_eZ;
}

void WLEMDMEG::setEz( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eZ = vec;
}

LaBP::WEGeneralCoilType::Enum WLEMDMEG::getChannelType( size_t channelId ) const
{
    WAssert( channelId < m_data->size(), "Index out of bounds!" );
    // Sequence: GGMGGMGGM ... 01 2 34 5
    if( channelId > 1 && ( channelId - 2 ) % 3 == 0 )
    {
        return LaBP::WEGeneralCoilType::MAGNETOMETER;
    }
    else
    {
        return LaBP::WEGeneralCoilType::GRADIOMETER;

    }
}
