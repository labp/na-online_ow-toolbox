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

#include <core/common/WLogger.h>

#include "WLTransformation.h"

const std::string WLTransformation::CLASS = "WLTransformation";

WLTransformation::WLTransformation( WLECoordSystem::Enum from, WLECoordSystem::Enum to, WLEUnit::Enum unit,
                WLEExponent::Enum exponent ) :
                m_from( from ), m_to( to ), m_unit( unit ), m_exponent( exponent )
{
    m_transformation.setZero(); // Set zero for empty(). Transformation matrix should never be zero due to homogeneous transl.
}

WLTransformation::WLTransformation( const WLTransformation& obj )
{
    m_from = obj.m_from;
    m_to = obj.m_to;
    m_unit = obj.m_unit;
    m_exponent = obj.m_exponent;
    m_transformation = obj.m_transformation;
}

WLTransformation::~WLTransformation()
{
}

WLTransformation::SPtr WLTransformation::instance()
{
    WLTransformation::SPtr instance( new WLTransformation() );
    return instance;
}

void WLTransformation::setIdentity()
{
    m_transformation.setIdentity();
}

void WLTransformation::data( const TransformationT& transformation )
{
    m_transformation = transformation;
}

WLTransformation::TransformationT& WLTransformation::data()
{
    return m_transformation;
}

const WLTransformation::TransformationT& WLTransformation::data() const
{
    return m_transformation;
}

WLTransformation::RotationT WLTransformation::rotation() const
{
    return m_transformation.block( 0, 0, 3, 3 );
}

WLTransformation::TranslationT WLTransformation::translation() const
{
    return m_transformation.block( 0, 3, 3, 1 );
}

bool WLTransformation::empty() const
{
    return m_transformation.isZero();
}

WLECoordSystem::Enum WLTransformation::from() const
{
    return m_from;
}

void WLTransformation::from( WLECoordSystem::Enum coordSystem )
{
    m_from = coordSystem;
}

WLECoordSystem::Enum WLTransformation::to() const
{
    return m_to;
}

void WLTransformation::to( WLECoordSystem::Enum coordSystem )
{
    m_to = coordSystem;
}

WLEUnit::Enum WLTransformation::unit() const
{
    return m_unit;
}

void WLTransformation::unit( WLEUnit::Enum unit )
{
    m_unit = unit;
}

WLEExponent::Enum WLTransformation::exponent() const
{
    return m_exponent;
}

void WLTransformation::exponent( WLEExponent::Enum exponent )
{
    m_exponent = exponent;
}

WLTransformation::SPtr WLTransformation::inverse() const
{
    WLTransformation::SPtr inverse = WLTransformation::instance();
    inverse->from( to() );
    inverse->to( from() );
    inverse->unit( unit() );
    inverse->exponent( exponent() );
    inverse->data( m_transformation.inverse() );
    return inverse;
}

bool WLTransformation::isUnitCompatible( const WLPositions& positions ) const
{
    if( m_unit != WLEUnit::UNKNOWN && positions.unit() != WLEUnit::UNKNOWN )
    {
        if( m_unit != positions.unit() )
        {
            wlog::debug( CLASS ) << "Units are not equals!";
            return false;
        }
        if( m_exponent != positions.exponent() )
        {
            wlog::debug( CLASS ) << "Exponents are not equals!";
            return false;
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Translation unit is not set and could not be checked: " << m_unit << "/" << positions.unit();
    }
    return true;
}

bool WLTransformation::isCompatible( const WLPositions& positions ) const
{
    if( ( m_from != WLECoordSystem::UNKNOWN && positions.coordSystem() != WLECoordSystem::UNKNOWN ) )
    {
        if( m_from != positions.coordSystem() )
        {
            wlog::debug( CLASS ) << "Coordinate systems are not equals!";
            return false;
        }
    }
    else
    {
        wlog::warn( CLASS ) << "Coordinate system is not set and could not be checked: " << m_from << "/"
                        << positions.coordSystem();
    }
    if( !isUnitCompatible( positions ) )
    {
        return false;
    }
    return true;
}

WLPositions::SPtr WLTransformation::operator*( const WLPositions& positions ) const
{
    if( !isCompatible( positions ) )
    {
        throw WPreconditionNotMet( "Position and transformation are not compatible! Check unit, exponent and coordSystem." );
    }

    WLPositions::SPtr out( new WLPositions( positions ) );
    out->data() = ( m_transformation * positions.data().colwise().homogeneous() ).block( 0, 0,
                    WLPositions::PositionsT::RowsAtCompileTime, positions.size() );
    if( m_to != WLECoordSystem::UNKNOWN )
    {
        out->coordSystem( m_to );
    }
    return out;
}
