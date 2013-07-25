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

#ifndef WLEADFIELDINTERPOLATION_H_
#define WLEADFIELDINTERPOLATION_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <mne/mne_forwardsolution.h>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLDataTypes.h"

/**
 * Interpolates a high resolution leadfield (HD leadfield) with K nearest neighbors of each sensors.
 */
class WLeadfieldInterpolation
{
public:
    typedef std::vector< WPosition > PositionsT;

    typedef boost::shared_ptr< PositionsT > PositionsSPtr;

    static const std::string CLASS;

    WLeadfieldInterpolation();
    virtual ~WLeadfieldInterpolation();

    /**
     * Reads the postions and leadfield matrix and converts/sets the value.
     */
    bool prepareHDLeadfield( MNELIB::MNEForwardSolution::ConstSPtr hdLeadfield );

    /**
     * Sets the positions of real sensors.
     */
    void setSensorPositions( PositionsSPtr posSensors );

    /**
     * Sets the positions of the HD leadfield sensors.
     */
    void setHDLeadfieldPosition( PositionsSPtr posHdLeadfield );

    /**
     * Sets the high resolution leadfield.
     */
    void setHDLeadfield( WLMatrix::SPtr leadfield );

    /**
     * Fills the input matrix with a new interpolated matrix.
     *
     * \param leadfield Matrix to be filled.
     *
     * \return true if successful.
     */
    bool interpolate( WLMatrix::SPtr leadfield );

    /**
     * Generates a leadfield matrix for testing purpose.
     *
     * \param sensors Sensors/rows of the matrix
     * \param sources Sources/cols of the matrix
     *
     * \return A sensors*sources matrix with random elements.
     */
    static WLMatrix::SPtr generateRandomLeadfield( size_t sensors, size_t sources );

private:
    static const int NEIGHBORS;

    struct NeighborsT
    {
        std::vector< int >* indexNeighbors;
        std::vector< float >* squareDistances;
    };

    PositionsSPtr m_posSensors;

    PositionsSPtr m_posHDLeadfield;

    WLMatrix::SPtr m_hdLeadfield;

    bool searchNearestNeighbor( std::vector< NeighborsT >* const neighbors, const PositionsT& searchPoints,
                    const PositionsT& inputPoints );
};

#endif  // WLEADFIELDINTERPOLATION_H_
