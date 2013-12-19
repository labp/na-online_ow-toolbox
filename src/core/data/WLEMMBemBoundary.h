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

#ifndef WLEMMBEMBOUNDARY_H_
#define WLEMMBEMBOUNDARY_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/enum/WLEExponent.h"
#include "core/data/enum/WLEUnit.h"
#include "WLEMMEnumTypes.h"

class WLEMMBemBoundary
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMMBemBoundary > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMMBemBoundary > ConstSPtr;

    WLEMMBemBoundary();
    ~WLEMMBemBoundary();

    WLArrayList< WPosition >::SPtr getVertex();
    WLArrayList< WPosition >::ConstSPtr getVertex() const;
    void setVertex( WLArrayList< WPosition >::SPtr vertex );

    WLEUnit::Enum getVertexUnit() const;
    void setVertexUnit( WLEUnit::Enum unit );

    WLEExponent::Enum getVertexExponent() const;
    void setVertexExponent( WLEExponent::Enum exponent );

    LaBP::WEBemType::Enum getBemType() const;
    void setBemType( LaBP::WEBemType::Enum exponent );

    WLArrayList< WVector3i >::SPtr getFaces();
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;
    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    float getConductivity() const;
    void setConductivity( float conductivity );

    WLEUnit::Enum getConductivityUnit() const;
    void setConductivityUnit( WLEUnit::Enum unit );

private:
    WLArrayList< WPosition >::SPtr m_vertex;

    WLEUnit::Enum m_vertexUnit;
    WLEExponent::Enum m_vertexExponent;
    LaBP::WEBemType::Enum m_bemType;

    WLArrayList< WVector3i >::SPtr m_faces;

    float m_conductivity;
    WLEUnit::Enum m_conductivityUnit;
};

std::ostream& operator<<( std::ostream &strm, const WLEMMBemBoundary& obj );

#endif /* WLEMMBEMBOUNDARY_H_ */
