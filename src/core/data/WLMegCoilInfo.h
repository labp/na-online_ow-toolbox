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

#ifndef WLMEGCOILINFO_H_
#define WLMEGCOILINFO_H_

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

/**
 * Information about MEG coil.
 *
 * \author pieloth
 * \ingroup data
 */
class WLMegCoilInfo
{
public:
    /**
     * Convenience typedef for a boost::shared_ptr< WLMegCoilInfo >
     */
    typedef boost::shared_ptr< WLMegCoilInfo > SPtr;

    /**
     * Convenience typedef for a  boost::shared_ptr< const WLMegCoilInfo >
     */
    typedef boost::shared_ptr< const WLMegCoilInfo > ConstSPtr;

    typedef Eigen::Vector3d PositionT;
    typedef Eigen::Matrix3Xd PositionsT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Vector3d OrientationT;  //!< Rows: x, y, z; Columns: channels
    typedef Eigen::VectorXd VectorT;
    typedef Eigen::Vector3d Vector3T;

    WLMegCoilInfo();
    virtual ~WLMegCoilInfo();

    PositionT position; //!< Coil position in device coords.
    OrientationT orientation; //!< Coil orientation in device coords.
    double area; //!< TODO Area of the coil.
    double windings; //!< TODO Windings of the coil.
    PositionsT integrationPoints; //!< Integration points in local coil coords.
    VectorT integrationWeights; //!< Weights of integration points.
    Vector3T ex; //!< x-axis vector of local coil coords. in device coords.
    Vector3T ey; //!< x-axis vector of local coil coords. in device coords.
    Vector3T ez; //!< x-axis vector of local coil coords. in device coords.
};

#endif  // WLMEGCOILINFO_H_
