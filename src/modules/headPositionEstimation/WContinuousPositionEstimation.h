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

#ifndef WCONTINUOUSPOSITIONESTIMATION_H_
#define WCONTINUOUSPOSITIONESTIMATION_H_

#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "math/WDownhillSimplexMethod.hpp"

/**
 * TODO(pieloth): documentation
 *
 * \author pieloth
 */
class WContinuousPositionEstimation: public WDownhillSimplexMethod< 15 >
{
public:
    static const std::string CLASS;

    typedef Eigen::Vector3d PositionT;
    typedef Eigen::Vector3d Vector3T;
    typedef Eigen::MatrixXd MatrixT;

    WContinuousPositionEstimation( const std::vector< PositionT >& sensPos, const std::vector< Vector3T >& sensOri );
    virtual ~WContinuousPositionEstimation();

    virtual double func( const PointT& x ) const;

    void setData( const MatrixT& data );

private:
    const std::vector< PositionT > m_sensPos;
    const std::vector< Vector3T > m_sensOri;

    MatrixT getSample( size_t coilIdx ) const;
    MatrixT m_data;

    /**
     * TODO
     *
     * \param dipPos
     * \param sensPos
     * \param sensPos
     * \return
     */
    static MatrixT computeLeadfield( const PositionT& dipPos, const std::vector< PositionT >& sensPos,
                    const std::vector< Vector3T >& sensOri );

    /**
     * Compute the magnetic flux density for a magnetic dipole with a fixed strength.
     *
     * \param dipPos Position of the dipole, i.e. HPI coil.
     * \param sensPos Position of the sensor, i.e. magnetometer.
     * \param sensOri Orientation of the sensor, i.e. magnetometer.
     * \return Leadfield vector containing x, y, z.
     */
    static Vector3T computeMagneticDipole( const PositionT& dipPos, const PositionT& sensPos, const Vector3T& sensOri );

};

#endif  // WCONTINUOUSPOSITIONESTIMATION_H_
