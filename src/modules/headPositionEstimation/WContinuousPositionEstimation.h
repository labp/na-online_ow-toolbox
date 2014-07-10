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
#include <iostream>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "cppmath/DownhillSimplexMethod.hpp"

/**
 * TODO(pieloth): documentation
 *
 * \author pieloth
 */
class WContinuousPositionEstimation: public cppmath::DownhillSimplexMethod< 6 >
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WContinuousPositionEstimation > SPtr;

    typedef Eigen::Matrix< double, 3, 1 > PointT;
    typedef Eigen::Matrix< double, 3, 1 > OrientationT;
    typedef Eigen::Matrix< double, 3, 1 > MomentT;
    typedef Eigen::Matrix< double, 4, 4 > TransformationT;
    typedef Eigen::MatrixXd MatrixT;

    WContinuousPositionEstimation( const std::vector< WPosition >& hpiPos, const std::vector< WPosition >& sensPos,
                    const std::vector< WVector3f >& sensOri );
    virtual ~WContinuousPositionEstimation();

    virtual double func( const ParamsT& x ) const;

    std::vector< WPosition > getResultPositions() const;

    TransformationT getResultTransformation() const;

    void nextSample();
    void setData( const MatrixT& data );

    ParamsT getInitialStep() const;

    void setInitialStep( const ParamsT& initial );

protected:
    virtual void createInitials( const ParamsT& initial );

private:
    typedef Eigen::Matrix< double, 4, 1 > HPointT;
    typedef Eigen::Matrix< double, 4, Eigen::Dynamic > HPointsT;
    typedef PointT Vector3T;
    typedef Eigen::Matrix3d RotationT;

    TransformationT paramsToTrans( const ParamsT& params ) const;

    HPointsT m_hpiPos;
    std::vector< PointT > m_sensPos;
    std::vector< OrientationT > m_sensOri;

    ParamsT m_initStep;

    MatrixT getSample( size_t coilIdx ) const;
    MatrixT m_data;
    MatrixT::Index m_smpIdx;

    /**
     * TODO
     *
     * \param dipPos
     * \param sensPos
     * \param sensPos
     * \return
     */
    static MatrixT computeLeadfield( const PointT& dipPos, const std::vector< PointT >& sensPos,
                    const std::vector< OrientationT >& sensOri );

    /**
     * Compute the magnetic flux density for a magnetic dipole with a fixed strength.
     *
     * \param dipPos Position of the dipole, i.e. HPI coil.
     * \param sensPos Position of the sensor, i.e. magnetometer.
     * \param sensOri Orientation of the sensor, i.e. magnetometer.
     * \return Leadfield vector containing x, y, z.
     */
    static Vector3T computeMagneticDipole( const PointT& dipPos, const PointT& sensPos, const OrientationT& sensOri );

};

inline std::ostream& operator<<( std::ostream &strm, const WContinuousPositionEstimation& est )
{
    strm << WContinuousPositionEstimation::CLASS << ": ";
    strm << "maxIterations=" << est.getMaximumIterations() << ", ";
    strm << "epsilon=" << est.getEpsilon() << ", ";
    strm << "coefficients=[" << est.getReflectionCoeff() << ", " << est.getContractionCoeff() << ", " << est.getExpansionCoeff()
                    << "], ";
    strm << "initFactor=" << est.getInitialFactor() << ", ";
    strm << "initStep=" << est.getInitialStep().transpose();
    return strm;
}

#endif  // WCONTINUOUSPOSITIONESTIMATION_H_
