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

#ifndef WHEADPOSITIONCORRECTION_H_
#define WHEADPOSITIONCORRECTION_H_

#include <string>

#include <Eigen/Core>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/emd/WLEMDHPI.h"
#include "core/data/emd/WLEMDMEG.h"

/**
 * Corrects MEG data to a reference head position.
 * \n
 * 1. Calculate the inverse solution for a simple dipole sphere at current position.\n
 * 2. Correct MEG data by computing the forward solution for the "simple source data"  at the reference position.\n
 * 3. Use corrected MEG data to perform a source reconstruction  with individual forward model at reference position. (external)
 *
 * \author pieloth
 * \see \cite Knoesche2002
 */
class WHeadPositionCorrection
{
public:
    static const std::string CLASS;

    WHeadPositionCorrection();
    virtual ~WHeadPositionCorrection();

    /**
     * Initializes and pre-computes all required data.
     *
     * \return True if algorithm is ready for computation.
     */
    bool init();

    /**
     * Checks if algorithm is initialized.
     *
     * \return True if algorithm is ready for computation.
     */
    bool isInitialzied() const;

    /**
     * Resets the algorithm.
     */
    void reset();

    /**
     * Corrects the MEG data to a reference head position.
     *
     * \param megOut Contains corrected MEG data.
     * \param meg MEG input data.
     * \param hpi Contains transformation aka continous head position.
     * \return True, if megOut contains the corrected data.
     */
    bool process( WLEMDMEG* const megOut, const WLEMDMEG& meg, const WLEMDHPI& hpi );

    /**
     * Sets the reference positions.
     *
     * \param pos The reference positions.
     */
    void setRefPosition( const WPosition& pos );

    /**
     * Sets the MEG positions and orientations.
     *
     * \param meg The MEG to extract positions and orientation.
     */
    void setMegPosAndOri( const WLEMDMEG& meg );

    /**
     * Sets the sphere radius.
     *
     * \param r Radius in meter.
     */
    void setSphereRadius( float r );

    /**
     * Sets the movement threshold.
     *
     * \param t Threshold.
     */
    void setMovementThreshold( float t );

private:
    friend class WHeadPositionCorrectionTest;

    typedef Eigen::ArrayXd ArrayT;
    typedef Eigen::Vector3d PositionT;
    typedef Eigen::Matrix3Xd PositionsT; //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Vector3d OrientationT; //!< Rows: x, y, z; Columns: channels
    typedef Eigen::Matrix3Xd OrientationsT; //!< Rows: x, y, z; Columns: channels
    typedef Eigen::MatrixXd MatrixT;

    /**
     * Generates a set of dipoles which are arranged in a sphere.
     *
     * \param pos Destination for the dipole positions.
     * \param ori Destination for the dipole orientations.
     * \param nDipoles Number of dipoles to generate.
     * \param c Center of the spheare
     * \param r Radius of the sphere in meter (default 0.07m = 7cm).
     * \return True if pos and ori contains the data of generated sphere.
     */
    bool generateDipoleSphere( PositionsT* const pos, OrientationsT* const ori, size_t nDipoles, float r = 0.07 ) const;

    /**
     * Computes the forward model of an electrical dipole for MEG.
     *
     * \param lf Destination for leadfield.
     * \param mPos MEG positions.
     * \param mOri MEG orientations.
     * \param dPos Dipole positions.
     * \param dOri Dipole orientations.
     * \return True if lf contains leadfield.
     */
    bool computeForward( MatrixT* const lf, const PositionsT& mPos, const OrientationsT& mOri, const PositionsT& dPos,
                    const OrientationsT& dOri ) const;

    /**
     * Computes the inverse operator.
     *
     * \param g Destination for inverse operator.
     * \param lf Leadfield.
     * \return True if g contains inverse operator.
     */
    bool computeInverseOperation( MatrixT* const g, const MatrixT& lf ) const;

    /**
     * Checks if movement is threshold exceeded.
     *
     * \param trans Transformation to check.
     * \return True if movement must be corrected.
     */
    bool checkMovementThreshold( const WLEMDHPI::TransformationT& trans );

    bool m_isInitialized; //!< Flag to force initialization.

    WPosition m_refPos; //!< Reference position to use.

    WLEMDHPI::TransformationT m_transExc; //!< Last exceeded transformation/position.

    MatrixT m_lfRef; //!< Forward model for reference position.
    MatrixT m_lfNow; //!< Forward model for current head position.
    MatrixT m_gNow; //!< Inverse operator for  current head position.

    PositionsT m_megPos; //!< MEG coil positions.
    OrientationsT m_megOri; //!< MEG coil orientations.

    PositionsT m_dipPos; //!< Dipole positions.
    OrientationsT m_dipOri; //!< Dipole orientations.

    float m_radius; //!< Radius for dipole sphere.

    float m_movementThreshold; //!< Indicates if a correction is needed.
};

#endif  // WHEADPOSITIONCORRECTION_H_
