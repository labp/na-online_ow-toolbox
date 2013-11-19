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

#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/exceptions/WNotFound.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLDataTypes.h"
#include "WLEMMEnumTypes.h"
#include "WLEMMSurface.h"
#include "WLEMMBemBoundary.h"

#include "WLEMMSubject.h"

using WLMatrix::MatrixT;
using namespace LaBP;

WLEMMSubject::WLEMMSubject() :
                m_weight( -1.0 ), m_sex( WESex::UNKNOWN ), m_height( -1.0 ), m_hand( WEHand::UNKNOWN )
{
    m_isotrak.reset( new std::vector< WVector3f >() );
    m_bemBoundaries.reset( new std::vector< WLEMMBemBoundary::SPtr >() );
}

WLEMMSubject::~WLEMMSubject()
{
}

std::string WLEMMSubject::getName()
{
    return m_name;
}
boost::shared_ptr< boost::gregorian::date > WLEMMSubject::getBirthday()
{
    return m_birthday;
}
WESex::Enum WLEMMSubject::getSex()
{
    return m_sex;
}
WEHand::Enum WLEMMSubject::getHand()
{
    return m_hand;
}
float WLEMMSubject::getHeight()
{
    return m_height;
}
float WLEMMSubject::getWeight()
{
    return m_weight;
}
std::string WLEMMSubject::getComment()
{
    return m_comment;
}
std::string WLEMMSubject::getHisId()
{
    return m_hisId;
}

void WLEMMSubject::setHeight( float height )
{
    m_height = height;
}
void WLEMMSubject::setWeight( float weight )
{
    m_weight = weight;
}
void WLEMMSubject::setComment( std::string comment )
{
    m_comment = comment;
}
void WLEMMSubject::setHisId( std::string hisId )
{
    m_hisId = hisId;
}

void WLEMMSubject::setSex( WESex::Enum sex )
{
    m_sex = sex;
}
void WLEMMSubject::setHand( WEHand::Enum hand )
{
    m_hand = hand;
}
void WLEMMSubject::setName( std::string name )
{
    m_name = name;
}

std::vector< WVector3f >& WLEMMSubject::getIsotrak()
{
    return *m_isotrak;
}

void WLEMMSubject::setIsotrak( boost::shared_ptr< std::vector< WVector3f > > isotrak )
{
    m_isotrak = isotrak;
}

WLEMMSurface& WLEMMSubject::getSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const
{
    if( m_surfaces.find( hemisphere ) != m_surfaces.end() )
    {
        return *( m_surfaces.find( hemisphere )->second );
    }
    else
    {
        throw WNotFound( "Requested surface not found!" );
    }
}

void WLEMMSubject::setSurface( boost::shared_ptr< WLEMMSurface > surface )
{
    m_surfaces[surface->getHemisphere()] = surface;
}

MatrixT& WLEMMSubject::getLeadfield( WEModalityType::Enum modality ) const
{
    if( m_leadfields.find( modality ) != m_leadfields.end() )
    {
        return *( m_leadfields.find( modality )->second );
    }
    else
    {
        throw WNotFound( "Requested leadfield not found!" );
    }
}

void WLEMMSubject::setLeadfield( WEModalityType::Enum modality, WLMatrix::SPtr leadfield )
{
    m_leadfields[modality] = leadfield;
}

std::vector< WLEMMBemBoundary::SPtr >& WLEMMSubject::getBemBoundaries() const
{
    return *m_bemBoundaries;
}

void WLEMMSubject::setBemBoundaries( boost::shared_ptr< std::vector< WLEMMBemBoundary::SPtr > > bemBoundaries )
{
    m_bemBoundaries = bemBoundaries;
}
