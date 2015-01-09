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

#ifndef MODULES_HEADPOSITIONCORRECTION_WMEGCOILINFORMATION_H_
#define MODULES_HEADPOSITIONCORRECTION_WMEGCOILINFORMATION_H_

#include <Eigen/Core>

/**
 * Information about MEG coil geometry.
 * \see \cite SrcModelling2008
 *
 * \author pieloth
 */
namespace WMegCoilInformation
{
    typedef Eigen::Vector3d PositionT;
    typedef Eigen::Matrix3Xd PositionsT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Vector3d OrientationT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Matrix3Xd OrientationsT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::VectorXd VectorT;

    typedef struct
    {
        PositionsT positions;  //!< 3D positions of sensors
        OrientationsT orientations;  //!< 3D directions of sensors
        VectorT areas;  //!< area of coils
        VectorT windings;  //!< windings of coils
        PositionsT integrationPoints;  //!< 3D positions of integration points
        VectorT integrationWeights;  //!< weights of integrations points
    } WMegCoils;

    /**
     * Accurate coil descriptions for type T1 planar gradiometer, 3012.
     *
     * \param megCoils
     */
    void neuromagCoil3012( WMegCoils* const megCoils );

    /**
     * Accurate coil descriptions for type T3 planar gradiometer, 3014.
     *
     * \param megCoils
     */
    void neuromagCoil3014( WMegCoils* const megCoils );

    /**
     * Accurate coil descriptions for type T1 magnetometer, 3022.
     *
     * \param megCoils
     */
    void neuromagCoil3022( WMegCoils* const megCoils );

    /**
     * Accurate coil descriptions for type T3 magnetometer, 3024.
     *
     * \param megCoils
     */
    void neuromagCoil3024( WMegCoils* const megCoils );
}

#endif  // MODULES_HEADPOSITIONCORRECTION_WMEGCOILINFORMATION_H_
