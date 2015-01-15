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

#ifndef MODULES_HEADPOSITIONCORRECTION_WMEGFORWARD_H_
#define MODULES_HEADPOSITIONCORRECTION_WMEGFORWARD_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "../../core/daqSystem/WLDaqNeuromag.h"
#include "core/data/WLMegCoilInfo.h"

/**
 * Computes the MEG forward model for a spherical dipole arrangement.
 * \see \cite Fingberg2003
 *
 * \author pieloth
 */
namespace WMegForward
{
    typedef Eigen::Vector3d PositionT;
    typedef Eigen::Matrix3Xd PositionsT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Vector3d OrientationT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Matrix3Xd OrientationsT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::VectorXd VectorT;
    typedef Eigen::Vector3d Vector3T;
    typedef Eigen::MatrixXd MatrixT;

    const std::string NSNAME = "WMegForward";

    /**
     * Computes the MEG forward model for a spherical dipole arrangement.
     * \see SimBio, analysis/source/ansimulatormegspheres_c.cpp
     *
     * \param lfOut
     * \param megSensor
     * \param dPos
     * \param dOri
     * \return
     */
    bool computeForward( MatrixT* const lfOut, const std::vector< WLMegCoilInfo::SPtr >& megSensors, const PositionsT& dPos,
                    const OrientationsT& dOri );

    double weberToTesla( const std::vector< WLMegCoilInfo::SPtr >& coilInfos );

    /**
     * Transfers local 3D integration coordinates to global 3D coordinates.
     *
     * \param ipOut Filled with integration points in global coords.
     * \param megCoilInfo Coil geometry information.
     * \return True if ipOut contains global 3D coords.
     */
    bool computeIntegrationPoints( PositionsT* ipOut, const WLMegCoilInfo& megCoilInfo );
} /* namespace WMegForward */

#endif  // MODULES_HEADPOSITIONCORRECTION_WMEGFORWARD_H_
