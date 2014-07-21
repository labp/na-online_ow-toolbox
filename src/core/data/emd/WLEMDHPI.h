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

/**
 * Data class for reconstructed HPI amplitudes.
 *
 * \author pieloth
 */

#ifndef WLEMDHPI_H_
#define WLEMDHPI_H_

#include <ostream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WDefines.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLDigPoint.h"
#include "core/container/WLList.h"
#include "WLEMData.h"

class WLEMDHPI: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDHPI > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDHPI > ConstSPtr;

    typedef Eigen::Matrix< double, 4, 4 > TransformationT;

    static const std::string CLASS;

    WLEMDHPI();

    explicit WLEMDHPI( const WLEMDHPI& hpi );

    virtual ~WLEMDHPI();

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
     * Extracts the channel positions of the HPI coils from the digitization points.
     *
     * \param digPoints digitization points
     * \return true, if channel positions found.
     */
    bool setChannelPositions3d( WLList< WLDigPoint >::ConstSPtr digPoints );

    /**
     * Gets the number of HPI coils.
     *
     * \return Number of HPI coils
     */
    WLChanNrT getNrHpiCoils() const;

    /**
     * Sets the number of HPI coils.
     *
     * \param count Number of HPI coils
     * \return true, if count matches data and positions.
     */
    bool setNrHpiCoils( WLChanNrT count );

    /**
     * Returns the estimated transformation matrices.
     *
     * \return Transformation matrices.
     */
    WLArrayList< TransformationT >::SPtr getTransformations();

    /**
     * Returns the estimated transformation matrices.
     *
     * \return Transformation matrices.
     */
    WLArrayList< TransformationT >::ConstSPtr getTransformations() const;

    /**
     * Sets estimated transformation matrices.
     *
     * \param trans Transformation matrices to set.
     */
    void setTransformations( WLArrayList< TransformationT >::SPtr trans );

private:
    WLArrayList< WPosition >::SPtr m_chanPos3d;

    WLArrayList< TransformationT >::SPtr m_transformations;

    WLChanNrT m_nrHpiCoils;
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMDHPI& obj )
{
    const WLEMData& emd = static_cast< const WLEMData& >( obj );
    strm << emd;
    strm << ", positions=" << obj.getChannelPositions3d()->size();
    return strm;
}

#endif  // WLEMDHPI_H_
