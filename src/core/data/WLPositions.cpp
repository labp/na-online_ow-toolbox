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

#include <core/common/exceptions/WOutOfBounds.h>

#include "WLPositions.h"

WLPositions::WLPositions( WLEUnit::Enum unit, WLEExponent::Enum exponent, WLECoordSystem::Enum coordSystem ) :
                m_unit( unit ), m_exponent( exponent ), m_coordSystem( coordSystem )
{
}

WLPositions::~WLPositions()
{
}

WLPositions::SPtr WLPositions::instance()
{
    WLPositions::SPtr instance( new WLPositions() );
    return instance;
}

WLPositions::PositionT WLPositions::at( PositionsT::Index i )
{
    if( i >= m_positions.cols() )
    {
        throw WOutOfBounds();
    }
    return m_positions.col( i );
}

const WLPositions::PositionT WLPositions::at( PositionsT::Index i ) const
{
    if( i >= m_positions.cols() )
    {
        throw WOutOfBounds();
    }
    return m_positions.col( i );
}

WLPositions::PositionsT& WLPositions::data()
{
    return m_positions;
}

const WLPositions::PositionsT& WLPositions::data() const
{
    return m_positions;
}

void WLPositions::data( const PositionsT& positions )
{
    m_positions = positions;
}

WLPositions::IndexT WLPositions::size() const
{
    return m_positions.cols();
}

void WLPositions::resize( WLPositions::IndexT nPos )
{
    m_positions.resize( 3, nPos );
}

bool WLPositions::empty() const
{
    return m_positions.cols() == 0;
}

WLEUnit::Enum WLPositions::unit() const
{
    return m_unit;
}

void WLPositions::unit( WLEUnit::Enum unit )
{
    m_unit = unit;
}

WLEExponent::Enum WLPositions::exponent() const
{
    return m_exponent;
}

void WLPositions::exponent( WLEExponent::Enum exponent )
{
    m_exponent = exponent;
}

WLECoordSystem::Enum WLPositions::coordSystem() const
{
    return m_coordSystem;
}

void WLPositions::coordSystem( WLECoordSystem::Enum coordSystem )
{
    m_coordSystem = coordSystem;
}
