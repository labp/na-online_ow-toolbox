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

#include "core/data/WLMatrixTypes.h"

#include "WDataSetEMMEnumTypes.h"
#include "WDataSetEMMSurface.h"
#include "WDataSetEMMBemBoundary.h"

#include "WDataSetEMMSubject.h"

LaBP::WDataSetEMMSubject::WDataSetEMMSubject()
{
    m_isotrak.reset( new std::vector< WVector3f >() );
    m_bemBoundaries.reset( new std::vector< WDataSetEMMBemBoundary::SPtr >() );
}

LaBP::WDataSetEMMSubject::~WDataSetEMMSubject()
{
    // TODO(kaehler): Auto-generated destructor stub
}

std::string LaBP::WDataSetEMMSubject::getName()
{
    return m_name;
}
boost::shared_ptr< boost::gregorian::date > LaBP::WDataSetEMMSubject::getBirthday()
{
    return m_birthday;
}
LaBP::WESex::Enum LaBP::WDataSetEMMSubject::getSex()
{
    return m_sex;
}
LaBP::WEHand::Enum LaBP::WDataSetEMMSubject::getHand()
{
    return m_hand;
}
float LaBP::WDataSetEMMSubject::getHeight()
{
    return m_height;
}
float LaBP::WDataSetEMMSubject::getWeight()
{
    return m_weight;
}
std::string LaBP::WDataSetEMMSubject::getComment()
{
    return m_comment;
}
std::string LaBP::WDataSetEMMSubject::getHisId()
{
    return m_hisId;
}

void LaBP::WDataSetEMMSubject::setHeight( float height )
{
    m_height = height;
}
void LaBP::WDataSetEMMSubject::setWeight( float weight )
{
    m_weight = weight;
}
void LaBP::WDataSetEMMSubject::setComment( std::string comment )
{
    m_comment = comment;
}
void LaBP::WDataSetEMMSubject::setHisId( std::string hisId )
{
    m_hisId = hisId;
}

void LaBP::WDataSetEMMSubject::setSex( LaBP::WESex::Enum sex )
{
    m_sex = sex;
}
void LaBP::WDataSetEMMSubject::setHand( LaBP::WEHand::Enum hand )
{
    m_hand = hand;
}
void LaBP::WDataSetEMMSubject::setName( std::string name )
{
    m_name = name;
}

std::vector< WVector3f >& LaBP::WDataSetEMMSubject::getIsotrak()
{
    return *m_isotrak;
}

void LaBP::WDataSetEMMSubject::setIsotrak( boost::shared_ptr< std::vector< WVector3f > > isotrak )
{
    m_isotrak = isotrak;
}

LaBP::WDataSetEMMSurface& LaBP::WDataSetEMMSubject::getSurface( WDataSetEMMSurface::Hemisphere::Enum hemisphere ) const
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

void LaBP::WDataSetEMMSubject::setSurface( boost::shared_ptr< WDataSetEMMSurface > surface )
{
    m_surfaces[surface->getHemisphere()] = surface;
}

LaBP::MatrixT& LaBP::WDataSetEMMSubject::getLeadfield( WEModalityType::Enum modality ) const
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

void LaBP::WDataSetEMMSubject::setLeadfield( WEModalityType::Enum modality, MatrixSPtr leadfield )
{
    m_leadfields[modality] = leadfield;
}

std::vector< LaBP::WDataSetEMMBemBoundary::SPtr >& LaBP::WDataSetEMMSubject::getBemBoundaries() const
{
    return *m_bemBoundaries;
}

void LaBP::WDataSetEMMSubject::setBemBoundaries(
                boost::shared_ptr< std::vector< boost::shared_ptr< LaBP::WDataSetEMMBemBoundary > > > bemBoundaries )
{
    m_bemBoundaries = bemBoundaries;
}
