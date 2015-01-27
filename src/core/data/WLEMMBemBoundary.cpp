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

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"

#include "WLEMMBemBoundary.h"

const std::string WLEMMBemBoundary::CLASS = "WLEMMBemBoundary";

WLEMMBemBoundary::WLEMMBemBoundary()
{
//    setVertexUnit( WLEUnit::NONE );
//    setVertexExponent( WLEExponent::UNKNOWN );
    setConductivityUnit( WLEUnit::NONE );
    setBemType( WLEBemType::UNDEFINED );

    m_vertex = WLPositions::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMMBemBoundary::~WLEMMBemBoundary()
{
}

WLPositions::SPtr WLEMMBemBoundary::getVertex()
{
    return m_vertex;
}

WLPositions::ConstSPtr WLEMMBemBoundary::getVertex() const
{
    return m_vertex;
}

void WLEMMBemBoundary::setVertex( WLPositions::SPtr vertex )
{
    m_vertex = vertex;
}

//WLEUnit::Enum WLEMMBemBoundary::getVertexUnit() const
//{
//    return m_vertexUnit;
//}
//
//void WLEMMBemBoundary::setVertexUnit( WLEUnit::Enum unit )
//{
//    m_vertexUnit = unit;
//}
//
//WLEExponent::Enum WLEMMBemBoundary::getVertexExponent() const
//{
//    return m_vertexExponent;
//}
//
//void WLEMMBemBoundary::setVertexExponent( WLEExponent::Enum exponent )
//{
//    m_vertexExponent = exponent;
//}

WLEBemType::Enum WLEMMBemBoundary::getBemType() const
{
    return m_bemType;
}

void WLEMMBemBoundary::setBemType( WLEBemType::Enum type )
{
    m_bemType = type;
}

WLArrayList< WVector3i >::SPtr WLEMMBemBoundary::getFaces()
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMMBemBoundary::getFaces() const
{
    return m_faces;
}

void WLEMMBemBoundary::setFaces( WLArrayList< WVector3i >::SPtr faces )
{
    m_faces = faces;
}

float WLEMMBemBoundary::getConductivity() const
{
    return m_conductivity;
}

void WLEMMBemBoundary::setConductivity( float conductivity )
{
    m_conductivity = conductivity;
}

WLEUnit::Enum WLEMMBemBoundary::getConductivityUnit() const
{
    return m_conductivityUnit;
}

void WLEMMBemBoundary::setConductivityUnit( WLEUnit::Enum unit )
{
    m_conductivityUnit = unit;
}
