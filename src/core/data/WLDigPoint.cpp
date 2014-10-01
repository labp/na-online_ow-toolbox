//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#include <string>

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
    m_kind = WLEPointType::fromFIFF( kind );
    m_ident = ident;
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
