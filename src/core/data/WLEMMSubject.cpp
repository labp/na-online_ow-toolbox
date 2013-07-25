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

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLDataTypes.h"
#include "WLEMMEnumTypes.h"
#include "WLEMMSurface.h"
#include "WLEMMBemBoundary.h"

#include "WLEMMSubject.h"

using WLMatrix::MatrixT;

LaBP::WLEMMSubject::WLEMMSubject()
{
    m_isotrak.reset( new std::vector< WVector3f >() );
    m_bemBoundaries.reset( new std::vector< WLEMMBemBoundary::SPtr >() );
}

LaBP::WLEMMSubject::~WLEMMSubject()
{
    // TODO(kaehler): Auto-generated destructor stub
}

std::string LaBP::WLEMMSubject::getName()
{
    return m_name;
}
boost::shared_ptr< boost::gregorian::date > LaBP::WLEMMSubject::getBirthday()
{
    return m_birthday;
}
LaBP::WESex::Enum LaBP::WLEMMSubject::getSex()
{
    return m_sex;
}
LaBP::WEHand::Enum LaBP::WLEMMSubject::getHand()
{
    return m_hand;
}
float LaBP::WLEMMSubject::getHeight()
{
    return m_height;
}
float LaBP::WLEMMSubject::getWeight()
{
    return m_weight;
}
std::string LaBP::WLEMMSubject::getComment()
{
    return m_comment;
}
std::string LaBP::WLEMMSubject::getHisId()
{
    return m_hisId;
}

void LaBP::WLEMMSubject::setHeight( float height )
{
    m_height = height;
}
void LaBP::WLEMMSubject::setWeight( float weight )
{
    m_weight = weight;
}
void LaBP::WLEMMSubject::setComment( std::string comment )
{
    m_comment = comment;
}
void LaBP::WLEMMSubject::setHisId( std::string hisId )
{
    m_hisId = hisId;
}

void LaBP::WLEMMSubject::setSex( LaBP::WESex::Enum sex )
{
    m_sex = sex;
}
void LaBP::WLEMMSubject::setHand( LaBP::WEHand::Enum hand )
{
    m_hand = hand;
}
void LaBP::WLEMMSubject::setName( std::string name )
{
    m_name = name;
}

std::vector< WVector3f >& LaBP::WLEMMSubject::getIsotrak()
{
    return *m_isotrak;
}

void LaBP::WLEMMSubject::setIsotrak( boost::shared_ptr< std::vector< WVector3f > > isotrak )
{
    m_isotrak = isotrak;
}

LaBP::WLEMMSurface& LaBP::WLEMMSubject::getSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const
{
    if( m_surfaces.find( hemisphere ) != m_surfaces.end() )
    {
        return *( m_surfaces.find( hemisphere )->second );
    }
    else
    {
        throw WException( "Requested surface not found!" );
    }
}

void LaBP::WLEMMSubject::setSurface( boost::shared_ptr< WLEMMSurface > surface )
{
    m_surfaces[surface->getHemisphere()] = surface;
}

MatrixT& LaBP::WLEMMSubject::getLeadfield( WEModalityType::Enum modality ) const
{
    if( m_leadfields.find( modality ) != m_leadfields.end() )
    {
        return *( m_leadfields.find( modality )->second );
    }
    else
    {
        throw WException( "Requested leadfield not found!" );
    }
}

void LaBP::WLEMMSubject::setLeadfield( WEModalityType::Enum modality, WLMatrix::SPtr leadfield )
{
    m_leadfields[modality] = leadfield;
}

std::vector< LaBP::WLEMMBemBoundary::SPtr >& LaBP::WLEMMSubject::getBemBoundaries() const
{
    return *m_bemBoundaries;
}

void LaBP::WLEMMSubject::setBemBoundaries(
                boost::shared_ptr< std::vector< boost::shared_ptr< LaBP::WLEMMBemBoundary > > > bemBoundaries )
{
    m_bemBoundaries = bemBoundaries;
}
