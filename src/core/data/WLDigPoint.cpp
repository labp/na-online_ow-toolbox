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

#include "WLDigPoint.h"

const std::string WLDigPoint::CLASS = "WLDigPoint";

WLDigPoint::WLDigPoint() :
                m_kind( WLEPointType::UNKNOWN ), m_ident( -1 )
{
}

WLDigPoint::WLDigPoint( const PointT& pos, WLEPointType::Enum kind, WLIdentT ident ) :
                m_point( pos ), m_kind( kind ), m_ident( ident )
{
}

WLDigPoint::WLDigPoint( const PointT& pos, WLFiffLib::kind_t kind, WLFiffLib::ident_t ident )
{
    m_point = pos;
    m_ident = ident;
    m_kind = WLEPointType::fromFIFF(kind);
}

WLDigPoint::~WLDigPoint()
{
}

WLEPointType::Enum WLDigPoint::getKind() const
{
    return m_kind;
}

void WLDigPoint::setKind( WLEPointType::Enum kind )
{
    m_kind = kind;
}

WLFiffLib::ident_t WLDigPoint::getIdent() const
{
    return m_ident;
}

void WLDigPoint::setIdent( WLFiffLib::ident_t ident )
{
    m_ident = ident;
}

const WLDigPoint::PointT& WLDigPoint::getPoint() const
{
    return m_point;
}

void WLDigPoint::setPoint( const PointT& pos )
{
    m_point = pos;
}

bool WLDigPoint::checkCardinal( WLECardinalPoint::Enum ident ) const
{
    return m_kind == WLEPointType::CARDINAL && ident == m_ident;
}

std::ostream& operator<<( std::ostream &strm, const WLDigPoint& obj )
{
    strm << obj.CLASS << ": kind=" << obj.getKind() << "; ident=" << obj.getIdent() << "; point=(" << obj.getPoint() << ")";
    return strm;
}
