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
#include <core/common/exceptions/WPreconditionNotMet.h>
#include <core/common/exceptions/WOutOfBounds.h>

#include "WLPositions.h"

const std::string WLPositions::CLASS = "WLPositions";

WLPositions::WLPositions( WLEUnit::Enum unit, WLEExponent::Enum exponent, WLECoordSystem::Enum coordSystem ) :
                m_unit( unit ), m_exponent( exponent ), m_coordSystem( coordSystem )
{
}

WLPositions::WLPositions( const WLPositions& pos )
{
    m_unit = pos.m_unit;
    m_exponent = pos.m_exponent;
    m_coordSystem = pos.m_coordSystem;
    m_positions.resize( PositionsT::RowsAtCompileTime, pos.size() );
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
    m_positions.resize( PositionsT::RowsAtCompileTime, nPos );
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

WLPositions& WLPositions::operator+=( const WLPositions& positions )
{
    if( positions.empty() )
    {
        return *this;
    }

    if( ( m_coordSystem != WLECoordSystem::UNKNOWN && positions.coordSystem() != WLECoordSystem::UNKNOWN ) )
    {
        if( m_coordSystem != positions.coordSystem() )
        {
            throw WPreconditionNotMet( "From coordinate system are not equals!" );
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Coordinate system is not set and could not be checked: " << m_coordSystem << "/"
                        << positions.coordSystem();
    }
    if( m_unit != WLEUnit::NONE && positions.unit() != WLEUnit::NONE )
    {
        if( m_unit != positions.unit() )
        {
            throw WPreconditionNotMet( "Units are not equals!" );
        }
        if( m_exponent != positions.exponent() )
        {
            throw WPreconditionNotMet( "Exponents are not equals!" );
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Unit is not set and could not be checked: " << m_unit << "/" << positions.unit();
    }

    const PositionsT old = this->data();
    this->resize( old.cols() + positions.size() );
    this->data().block( 0, 0, PositionT::RowsAtCompileTime, old.cols() ) = old;
    this->data().block( 0, old.cols(), PositionT::RowsAtCompileTime, positions.size() ) = positions.data();

    return *this;
}
