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

#ifndef WLEMMSURFACE_H_
#define WLEMMSURFACE_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLPositions.h"

/**
 * Brain surface/model.
 *
 * \author pieloth
 * \ingroup data
 */
class WLEMMSurface
{
public:
    static const std::string CLASS;

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
    WLEMMSurface( WLPositions::SPtr vertex, WLArrayList< WVector3i >::SPtr faces, Hemisphere::Enum hemisphere );

    explicit WLEMMSurface( const WLEMMSurface& surface );

    ~WLEMMSurface();

    WLPositions::SPtr getVertex();
    WLPositions::ConstSPtr getVertex() const;
    void setVertex( WLPositions::SPtr vertex );

    WLArrayList< WVector3i >::SPtr getFaces();
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;
    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    Hemisphere::Enum getHemisphere() const;
    void setHemisphere( Hemisphere::Enum val );

private:
    WLPositions::SPtr m_vertex;
    WLArrayList< WVector3i >::SPtr m_faces;
    Hemisphere::Enum m_hemisphere;
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMMSurface& obj )
{
    strm << WLEMMSurface::CLASS << ": hemisphere=" << obj.getHemisphere();
    strm << ", vertices=" << obj.getVertex()->size();
    strm << ", faces=" << obj.getFaces()->size();
    return strm;
}

#endif  // WLEMMSURFACE_H_
