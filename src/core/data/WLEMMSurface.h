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

#ifndef WLEMMSURFACE_H_
#define WLEMMSURFACE_H_

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"

#include "WLEMMEnumTypes.h"

class WLEMMSurface
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMMSurface > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMMSurface > ConstSPtr;

    struct Hemisphere
    {
        enum Enum
        {
            LEFT, RIGHT, BOTH
        };
    };

    WLEMMSurface();
    WLEMMSurface( WLArrayList< WPosition >::SPtr vertex, LaBP::WEUnit::Enum vertexUnit, LaBP::WEExponent::Enum vertexExponent,
                    WLArrayList< WVector3i >::SPtr faces, Hemisphere::Enum hemisphere );

    WLEMMSurface( const WLEMMSurface& surface );

    ~WLEMMSurface();

    WLArrayList< WPosition >::SPtr getVertex();
    WLArrayList< WPosition >::ConstSPtr getVertex() const;
    void setVertex( WLArrayList< WPosition >::SPtr vertex );

    LaBP::WEUnit::Enum getVertexUnit() const;
    void setVertexUnit( LaBP::WEUnit::Enum unit );

    LaBP::WEExponent::Enum getVertexExponent() const;
    void setVertexExponent( LaBP::WEExponent::Enum exponent );

    WLArrayList< WVector3i >::SPtr getFaces();
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;
    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    Hemisphere::Enum getHemisphere() const;
    void setHemisphere( Hemisphere::Enum val );

private:
    WLArrayList< WPosition >::SPtr m_vertex;

    Hemisphere::Enum m_hemisphere;

    LaBP::WEUnit::Enum m_vertexUnit;
    LaBP::WEExponent::Enum m_vertexExponent;

    WLArrayList< WVector3i >::SPtr m_faces;
};

#endif  // WLEMMSURFACE_H_
