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

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"

#include "WLEMMEnumTypes.h"
#include "WLEMMBemBoundary.h"

using namespace LaBP;

const std::string WLEMMBemBoundary::CLASS = "WLEMMBemBoundary";

WLEMMBemBoundary::WLEMMBemBoundary()
{
    setVertexUnit( WLEUnit::NONE );
    setVertexExponent( WLEExponent::UNKNOWN );

    setConductivityUnit( WLEUnit::NONE );

    m_vertex = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMMBemBoundary::~WLEMMBemBoundary()
{
}

WLArrayList< WPosition >::SPtr WLEMMBemBoundary::getVertex()
{
    return m_vertex;
}

WLArrayList< WPosition >::ConstSPtr WLEMMBemBoundary::getVertex() const
{
    return m_vertex;
}

void WLEMMBemBoundary::setVertex( WLArrayList< WPosition >::SPtr vertex )
{
    m_vertex = vertex;
}

WLEUnit::Enum WLEMMBemBoundary::getVertexUnit() const
{
    return m_vertexUnit;
}

void WLEMMBemBoundary::setVertexUnit( WLEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

WLEExponent::Enum WLEMMBemBoundary::getVertexExponent() const
{
    return m_vertexExponent;
}

void WLEMMBemBoundary::setVertexExponent( WLEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

WEBemType::Enum WLEMMBemBoundary::getBemType() const
{
    return m_bemType;
}

void WLEMMBemBoundary::setBemType( WEBemType::Enum type )
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

std::ostream& operator<<( std::ostream &strm, const WLEMMBemBoundary& obj )
{
    strm << WLEMMBemBoundary::CLASS << ": type=" << WEBemType::name( obj.getBemType() );
    strm << ", vertices=" << obj.getVertex()->size();
    strm << ", faces=" << obj.getFaces()->size();
    return strm;
}
