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
                m_kind( PointType::UNKNOWN ), m_ident( -1 )
{
}

WLDigPoint::WLDigPoint( const WPosition& pos, PointType::Enum kind, int ident ) :
                m_point( pos ), m_kind( kind ), m_ident( ident )
{
}

WLDigPoint::WLDigPoint( const WPosition& pos, int kind, int ident )
{
    m_point = pos;
    m_ident = ident;
    // Note: Change this when adding a new enum entry.
    switch( kind )
    {
        case 1:
            m_kind = PointType::CARDINAL;
            break;
        case 2:
            m_kind = PointType::HPI;
            break;
        case 3:
            // Note: Could be ECG
            m_kind = PointType::EEG;
            wlog::debug( CLASS ) << "Kind could be EEG or ECG. Using EEG!";
            break;
        case 4:
            m_kind = PointType::EXTRA;
            break;
        case 5:
            m_kind = PointType::HEAD_SURFACE;
            break;
        default:
            wlog::warn( CLASS ) << "Unknown kind!";
            m_kind = PointType::UNKNOWN;
            break;
    }
}

WLDigPoint::~WLDigPoint()
{
}

WLDigPoint::PointType::Enum WLDigPoint::getKind() const
{
    return m_kind;
}

void WLDigPoint::setKind( PointType::Enum kind )
{
    m_kind = kind;
}

int WLDigPoint::getIdent() const
{
    return m_ident;
}

void WLDigPoint::setIdent( int ident )
{
    m_ident = ident;
}

const WPosition& WLDigPoint::getPoint() const
{
    return m_point;
}

void WLDigPoint::setPoint( const WPosition& pos )
{
    m_point = pos;
}

bool WLDigPoint::checkCardinal( CardinalPoints::Enum ident ) const
{
    return m_kind == PointType::CARDINAL && ident == m_ident;
}
