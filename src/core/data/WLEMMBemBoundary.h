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

#ifndef WLEMMBEMBOUNDARY_H_
#define WLEMMBEMBOUNDARY_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLPositions.h"
#include "core/data/enum/WLEExponent.h"
#include "core/data/enum/WLEBemType.h"
#include "core/data/enum/WLEUnit.h"

/**
 * Boundary Element Model of a subject.
 *
 * \author pieloth
 * \ingroup data
 */
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

    WLPositions::SPtr getVertex();
    WLPositions::ConstSPtr getVertex() const;
    void setVertex( WLPositions::SPtr vertex );

    WLEBemType::Enum getBemType() const;
    void setBemType( WLEBemType::Enum exponent );

    WLArrayList< WVector3i >::SPtr getFaces();
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;
    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    float getConductivity() const;
    void setConductivity( float conductivity );

    WLEUnit::Enum getConductivityUnit() const;
    void setConductivityUnit( WLEUnit::Enum unit );

private:
    WLPositions::SPtr m_vertex;

    WLEUnit::Enum m_vertexUnit;
    WLEExponent::Enum m_vertexExponent;
    WLEBemType::Enum m_bemType;

    WLArrayList< WVector3i >::SPtr m_faces;

    float m_conductivity;
    WLEUnit::Enum m_conductivityUnit;
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMMBemBoundary& obj )
{
    strm << WLEMMBemBoundary::CLASS << ": type=" << obj.getBemType();
    strm << ", vertices=" << obj.getVertex()->size();
    strm << ", faces=" << obj.getFaces()->size();
    return strm;
}

#endif  // WLEMMBEMBOUNDARY_H_
