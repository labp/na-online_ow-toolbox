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
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMMSurface.h"

const std::string WLEMMSurface::CLASS = "WLEMMSurface";

WLEMMSurface::WLEMMSurface()
{
    m_vertex = WLPositions::instance();
    m_faces = WLArrayList< WVector3i >::instance();
    m_hemisphere = Hemisphere::BOTH;
}

WLEMMSurface::WLEMMSurface( WLPositions::SPtr vertex, WLArrayList< WVector3i >::SPtr faces, Hemisphere::Enum hemisphere ) :
                m_vertex( vertex ), m_faces( faces ), m_hemisphere( hemisphere )
{
}

WLEMMSurface::WLEMMSurface( const WLEMMSurface& surface )
{
    m_vertex = WLPositions::instance();
    m_faces = WLArrayList< WVector3i >::instance();
    m_hemisphere = surface.m_hemisphere;
}

WLEMMSurface::~WLEMMSurface()
{
}

WLPositions::SPtr WLEMMSurface::getVertex()
{
    return m_vertex;
}

WLPositions::ConstSPtr WLEMMSurface::getVertex() const
{
    return m_vertex;
}

void WLEMMSurface::setVertex( WLPositions::SPtr vertex )
{
    m_vertex = vertex;
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
