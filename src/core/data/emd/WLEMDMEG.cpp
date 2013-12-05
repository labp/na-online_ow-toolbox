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

using namespace LaBP;

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

WLEMDMEG::DataSPtr WLEMDMEG::getData( WEGeneralCoilType::Enum type ) const
{
    if( getNrChans() % 3 != 0 )
    {
        return WLEMDMEG::DataSPtr( new WLEMDMEG::DataT );
    }

    std::vector< size_t > picks = getPicks( type );
    WLEMDMEG::DataSPtr dataPtr( new WLEMDMEG::DataT( picks.size(), getSamplesPerChan() ) );
    WLEMDMEG::DataT& data = *dataPtr;

    size_t row = 0;
    std::vector< size_t >::const_iterator it;
    for( it = picks.begin(); it != picks.end(); ++it )
    {
        data.row( row++ ) = m_data->row( *it );
    }

    return dataPtr;
}

std::vector< size_t > WLEMDMEG::getPicks( WEGeneralCoilType::Enum type ) const
{
    if( m_picksMag.size() + m_picksGrad.size() != getNrChans() )
    {
        m_picksGrad.clear();
        m_picksMag.clear();
        const size_t rows = getNrChans();
        if( rows % 3 != 0 )
        {
            return std::vector< size_t >(); // empty vector
        }

        m_picksMag.reserve( rows / 3 );
        m_picksMag.reserve( ( rows / 3 ) * 2 );

        for( size_t ch = 0; ch < rows; ++ch )
        {
            switch( getChannelType( ch ) )
            {
                case WEGeneralCoilType::MAGNETOMETER:
                    m_picksMag.push_back( ch );
                    break;
                case WEGeneralCoilType::GRADIOMETER:
                    m_picksGrad.push_back( ch );
                    break;
            }
        }
    }

    switch( type )
    {
        case WEGeneralCoilType::MAGNETOMETER:
            return m_picksMag;
        case WEGeneralCoilType::GRADIOMETER:
            return m_picksGrad;
        default:
            return std::vector< size_t >(); // empty vector
    }
}

std::ostream& operator<<( std::ostream &strm, const WLEMDMEG& obj )
{
    const WLEMData& emd = static_cast< const WLEMData& >( obj );
    strm << emd;
    strm << ", positions=" << obj.getChannelPositions3d()->size();
    strm << ", faces=" << obj.getFaces()->size();
    return strm;
}
