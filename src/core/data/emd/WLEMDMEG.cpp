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
    m_chanPos3d = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();

    m_eX = WLArrayList< WVector3f >::instance();
    m_eY = WLArrayList< WVector3f >::instance();
    m_eZ = WLArrayList< WVector3f >::instance();
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

WLArrayList< WPosition >::SPtr WLEMDMEG::getChannelPositions3d()
{
    return m_chanPos3d;
}

WLArrayList< WPosition >::ConstSPtr WLEMDMEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void WLEMDMEG::setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d )
{
    m_chanPos3d = chanPos3d;
}

void WLEMDMEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    m_chanPos3d = WLArrayList< WPosition >::instance( *chanPos3d );
}

WLArrayList< WVector3i >::SPtr WLEMDMEG::getFaces()
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMDMEG::getFaces() const
{
    return m_faces;
}

void WLEMDMEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = WLArrayList< WVector3i >::instance( *faces );
}

void WLEMDMEG::setFaces( WLArrayList< WVector3i >::SPtr faces )
{
    m_faces = faces;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEx()
{
    return m_eX;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEx() const
{
    return m_eX;
}

void WLEMDMEG::setEx( WLArrayList< WVector3f >::SPtr vec )
{
    m_eX = vec;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEy()
{
    return m_eY;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEy() const
{
    return m_eY;
}

void WLEMDMEG::setEy( WLArrayList< WVector3f >::SPtr vec )
{
    m_eY = vec;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEz()
{
    return m_eZ;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEz() const
{
    return m_eZ;
}

void WLEMDMEG::setEz( WLArrayList< WVector3f >::SPtr vec )
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
