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

#include "WLEMMSubject.h"

#include <boost/shared_ptr.hpp>
#include <core/common/exceptions/WNotFound.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/data/enum/WLEModality.h"

#include "WLDataTypes.h"
#include "WLEMMBemBoundary.h"
#include "WLEMMSurface.h"

using WLMatrix::MatrixT;

const std::string WLEMMSubject::CLASS = "WLEMMSubject";

WLEMMSubject::WLEMMSubject()
{
    m_isotrak = WLArrayList< WVector3f >::instance();
    m_bemBoundaries = WLList< WLEMMBemBoundary::SPtr >::instance();
}

WLEMMSubject::~WLEMMSubject()
{
}

std::string WLEMMSubject::getName()
{
    return m_name;
}

std::string WLEMMSubject::getComment()
{
    return m_comment;
}
std::string WLEMMSubject::getHisId()
{
    return m_hisId;
}

void WLEMMSubject::setComment( std::string comment )
{
    m_comment = comment;
}

void WLEMMSubject::setHisId( std::string hisId )
{
    m_hisId = hisId;
}

void WLEMMSubject::setName( std::string name )
{
    m_name = name;
}

WLArrayList< WVector3f >::SPtr WLEMMSubject::getIsotrak()
{
    return m_isotrak;
}

WLArrayList< WVector3f >::ConstSPtr WLEMMSubject::getIsotrak() const
{
    return m_isotrak;
}

void WLEMMSubject::setIsotrak( WLArrayList< WVector3f >::SPtr isotrak )
{
    m_isotrak = isotrak;
}

WLEMMSurface::SPtr WLEMMSubject::getSurface( WLEMMSurface::Hemisphere::Enum hemisphere )
{
    if( m_surfaces.find( hemisphere ) != m_surfaces.end() )
    {
        return m_surfaces.find( hemisphere )->second;
    }
    else
    {
        throw WNotFound( "Requested surface not found!" );
    }
}

WLEMMSurface::ConstSPtr WLEMMSubject::getSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const
{
    // FIX(pieloth): Do not call getSurface to reduce redundant code -> atom loop.
    if( m_surfaces.find( hemisphere ) != m_surfaces.end() )
    {
        return m_surfaces.find( hemisphere )->second;
    }
    else
    {
        throw WNotFound( "Requested surface not found!" );
    }
}

void WLEMMSubject::setSurface( WLEMMSurface::SPtr surface )
{
    m_surfaces[surface->getHemisphere()] = surface;
}

WLMatrix::SPtr WLEMMSubject::getLeadfield( WLEModality::Enum modality )
{
    if( m_leadfields.find( modality ) != m_leadfields.end() )
    {
        return m_leadfields.find( modality )->second;
    }
    else
    {
        throw WNotFound( "Requested leadfield not found!" );
    }
}

WLMatrix::ConstSPtr WLEMMSubject::getLeadfield( WLEModality::Enum modality ) const
{
    // FIX(pieloth): Do not call getLeadfield to reduce redundant code -> atom loop.
    if( m_leadfields.find( modality ) != m_leadfields.end() )
    {
        return m_leadfields.find( modality )->second;
    }
    else
    {
        throw WNotFound( "Requested leadfield not found!" );
    }
}

void WLEMMSubject::setLeadfield( WLEModality::Enum modality, WLMatrix::SPtr leadfield )
{
    m_leadfields[modality] = leadfield;
}

WLList< WLEMMBemBoundary::SPtr >::SPtr WLEMMSubject::getBemBoundaries()
{
    return m_bemBoundaries;
}

WLList< WLEMMBemBoundary::SPtr >::ConstSPtr WLEMMSubject::getBemBoundaries() const
{
    return m_bemBoundaries;
}

void WLEMMSubject::setBemBoundaries( WLList< WLEMMBemBoundary::SPtr >::SPtr bemBoundaries )
{
    m_bemBoundaries = bemBoundaries;
}

std::ostream& operator<<( std::ostream &strm, const WLEMMSubject& obj )
{
    strm << WLEMMSubject::CLASS << ": ";
    strm << "surface[" << WLEMMSurface::Hemisphere::LEFT << "]=" << obj.hasSurface( WLEMMSurface::Hemisphere::LEFT );
    strm << ", surface[" << WLEMMSurface::Hemisphere::RIGHT << "]=" << obj.hasSurface( WLEMMSurface::Hemisphere::RIGHT );
    strm << ", leadfield[EEG]=" << obj.hasLeadfield( WLEModality::EEG );
    strm << ", leadfield[MEG]=" << obj.hasLeadfield( WLEModality::MEG );
    strm << ", BEMs=" << obj.getBemBoundaries()->size();
    strm << ", isotrak=" << obj.getIsotrak()->size();
    return strm;
}
