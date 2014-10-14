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

#ifndef WLEMDEEG_H
#define WLEMDEEG_H

#include <ostream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WDefines.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"

#include "WLEMData.h"

class WLEMDEEG: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDEEG > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDEEG > ConstSPtr;

    WLEMDEEG();

    explicit WLEMDEEG( const WLEMDEEG& eeg );

    virtual ~WLEMDEEG();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    /**
     * Returns the positions in millimeter.
     */
    WLArrayList< WPosition >::SPtr getChannelPositions3d();

    /**
     * Returns the positions in millimeter.
     */
    WLArrayList< WPosition >::ConstSPtr getChannelPositions3d() const;

    /**
     * Sets the positions. Positions must be in millimeter.
     */
    void setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d );

    OW_API_DEPRECATED
    void setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d );

    /**
     * Returns the faces.
     */
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;

    WLArrayList< WVector3i >::SPtr getFaces();

    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    OW_API_DEPRECATED
    void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

private:
    WLArrayList< WPosition >::SPtr m_chanPos3d;

    WLArrayList< WVector3i >::SPtr m_faces;
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMDEEG& obj )
{
    const WLEMData& emd = static_cast< const WLEMData& >( obj );
    strm << emd;
    strm << ", positions=" << obj.getChannelPositions3d()->size();
    strm << ", faces=" << obj.getFaces()->size();
    return strm;
}

#endif  // WLEMDEEG_H
