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

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMMEnumTypes.h"
#include "WLEMMSurface.h"

using namespace LaBP;

WLEMMSurface::WLEMMSurface()
{
    setVertexUnit( WEUnit::UNKNOWN_UNIT );
    setVertexExponent( WEExponent::BASE );

    m_vertex = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMMSurface::WLEMMSurface( WLArrayList< WPosition >::SPtr vertex, WEUnit::Enum vertexUnit, WEExponent::Enum vertexExponent,
                WLArrayList< WVector3i >::SPtr faces, Hemisphere::Enum hemisphere ) :
                m_vertex( vertex ), m_vertexUnit( vertexUnit ), m_vertexExponent( vertexExponent ), m_faces( faces ), m_hemisphere(
                                hemisphere )
{
}

WLEMMSurface::WLEMMSurface( const WLEMMSurface& surface )
{
    m_hemisphere = surface.m_hemisphere;
    m_vertexUnit = surface.m_vertexUnit;
    m_vertexExponent = surface.m_vertexExponent;

    m_vertex = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMMSurface::~WLEMMSurface()
{
}

WLArrayList< WPosition >::SPtr WLEMMSurface::getVertex()
{
    return m_vertex;
}

WLArrayList< WPosition >::ConstSPtr WLEMMSurface::getVertex() const
{
    return m_vertex;
}

void WLEMMSurface::setVertex( WLArrayList< WPosition >::SPtr vertex )
{
    m_vertex = vertex;
}

WEUnit::Enum WLEMMSurface::getVertexUnit() const
{
    return m_vertexUnit;
}

void WLEMMSurface::setVertexUnit( WEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

WEExponent::Enum WLEMMSurface::getVertexExponent() const
{
    return m_vertexExponent;
}

void WLEMMSurface::setVertexExponent( WEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

WLArrayList< WVector3i >::SPtr WLEMMSurface::getFaces()
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMMSurface::getFaces() const
{
    return m_faces;
}

void WLEMMSurface::setFaces( WLArrayList< WVector3i >::SPtr faces )
{
    m_faces = faces;
}

WLEMMSurface::Hemisphere::Enum WLEMMSurface::getHemisphere() const
{
    return m_hemisphere;
}

void WLEMMSurface::setHemisphere( Hemisphere::Enum val )
{
    m_hemisphere = val;
}
