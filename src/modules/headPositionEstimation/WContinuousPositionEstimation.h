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

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLPositions.h"
#include "core/data/WLTransformation.h"
#include "cppmath/DownhillSimplexMethod.hpp"

/**
 * Estimates the positions of the HPI coils in MEG coordinates by using a rigid transformation,
 * Nedler-Mead method for optimization and magnetic dipoles with fixed strengths for forward problem.
 * \see \cite Stolk2013
 * \see FieldTrip Project, ft_realtime_headlocalizer
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
    typedef Eigen::MatrixXd MatrixT;

    /**
     * Constructor.
     *
     * \param hpiPos HPI positions in head coordinate system and meter.
     * \param sensPos Sensor positions, i.e. magnetometer positions, in device coordinate system and meter.
     * \param sensOri Sensor orientation, i.e. magnetometer orientation, in device coordinate system.
     */
    WContinuousPositionEstimation( const WLPositions& hpiPos, const WLPositions& sensPos,
                    const std::vector< WVector3f >& sensOri );
    virtual ~WContinuousPositionEstimation();

    /**
     * Computes the fit error between the reconstructed magnetic flux density after transforming the HPI coils
     * and the actual measured data.
     *
     * \param x Parameter, i.e. rotation (radiant) and translation (meter).
     * \return fit error
     */
    virtual double func( const ParamsT& x ) const;

    /**
     * Calculates the transformed positions of the HPI coils.
     *
     * \return HPI coil positions in MEG device coordinates.
     */
    WLPositions::SPtr getResultPositions() const;

    /**
     * Gets the final transformation matrix from head coordinates to MEG device coordinates.
     *
     * \return Transformation matrix.
     */
    WLTransformation::SPtr getResultTransformation() const;

    /**
     * Move index to next data samples.
     */
    void nextSample();

    /**
     * Sets the data, i.e. extracted signals for each HPI coil on each magnetometer coil.
     *
     * \param data
     */
    void setData( const MatrixT& data );

private:
    typedef Eigen::Matrix< double, 4, 1 > HPointT; /**< Homogeneous point for matrix operations. */
    typedef Eigen::Matrix< double, 4, Eigen::Dynamic > HPointsT; /**< Array of homogeneous points for matrix operations. */
    typedef PointT Vector3T;
    typedef Eigen::Matrix3d RotationT;
    typedef WLTransformation::TransformationT TransformationT;

    const WLEUnit::Enum m_unit;
    const WLEExponent::Enum m_exponent;

    /**
     * Calculates the transformation matrix from the rotation (radiant) and translation (meter).
     *
     * \param params [rx, ry, rz, tx, ty, tz]
     * \return Transformation matrix using z-y-x rotation.
     */
    TransformationT paramsToTrans( const ParamsT& params ) const;

    HPointsT m_hpiPos; //!< In meter as translation.
    std::vector< PointT > m_sensPos; //!< In meter as translation.
    std::vector< OrientationT > m_sensOri;

    /**
     * Extracts the data for a HPI coil for the current sample.
     *
     * \param coilIdx Index of HPI coil.
     * \return Sample for HPI coil and current sample.
     */
    MatrixT getSample( size_t coilIdx ) const;
    MatrixT m_data;
    MatrixT::Index m_smpIdx;

    /**
     * Computes the leadfield for the HPI coils to MEG coils/magnetometer.
     *
     * \param dipPos Position of the dipole, i.e. HPI coil.
     * \param sensPos Positions of the sensors, i.e. magnetometers.
     * \param sensOri Orientations of the sensors, i.e. magnetometers.
     * \return Leadfield matrix for dipole to sensors (row: MEG coil; col: x, y, z).
     */
    static MatrixT computeLeadfield( const PointT& dipPos, const std::vector< PointT >& sensPos,
                    const std::vector< OrientationT >& sensOri );

    /**
     * Computes the magnetic flux density for a magnetic dipole with a fixed strength.
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
    return strm;
}

#endif  // WCONTINUOUSPOSITIONESTIMATION_H_
