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

#ifndef WEEGSKINALIGNMENT_H_
#define WEEGSKINALIGNMENT_H_

#include <string>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLEMMeasurement.h"

#include "WAlignment.h"

/**
 * Aligns the EEG sensor positions to the BEM skin layer
 * by using the fiducial points (lpa, nasion, rpa) from both coordinate systems as correspondences.
 * Note: Correspondences for WAlignment are set by this implementation!
 *
 * \author pieloth
 */
class WEEGSkinAlignment: public WAlignment
{
public:
    static const std::string CLASS;

    /**
     * Constructor.
     *
     * \param maxInterations Maximum iterations for ICP.
     */
    explicit WEEGSkinAlignment( int maxInteration = 10 );
    virtual ~WEEGSkinAlignment();

    /**
     * Computes the alignment and stores the transformation matrix.
     * Extracts the fiducial points for EEG from the digitization points of the measurement.
     * Fiducial points for BEM skin layer must set manually.
     *
     * \param matrix Holds the final transformation.
     * \param emm EM measurement containing digitization points, EEG sensor positions and a BEM skin layer.
     * \return Fitness score (>=0) if converged or NOT_CONVERGED.
     */
    double align( TransformationT* const matrix, WLEMMeasurement::ConstSPtr emm );

    /**
     * Returns the left pre-auricular position on the BEM skin layer.
     *
     * \return the position.
     */
    const WPosition& getLpaSkin() const;

    /**
     * Sets the left pre-auricular position on the BEM skin layer.
     *
     * \param lpaSkin the position.
     */
    void setLpaSkin( const WPosition& lpaSkin );

    /**
     * Returns the nasion position on the BEM skin layer.
     *
     * \return the position.
     */
    const WPosition& getNasionSkin() const;

    /**
     * Sets the nasion position on the BEM skin layer.
     *
     * \param nasionSkin the position.
     */
    void setNasionSkin( const WPosition& nasionSkin );

    /**
     * Returns the right pre-auricular position on the BEM skin layer.
     *
     * \return the position.
     */
    const WPosition& getRpaSkin() const;

    /**
     * Sets the right pre-auricular position on the BEM skin layer.
     *
     * \param rpaSkin the position.
     */
    void setRpaSkin( const WPosition& rpaSkin );

private:
    bool extractFiducialPoints( WPosition* const lpa, WPosition* const nasion, WPosition* const rpa, const WLEMMeasurement& emm );

    bool extractBEMSkinPoints( PointsT* const out, const WLEMMeasurement& emm );

    WPosition m_lpaSkin;
    WPosition m_nasionSkin;
    WPosition m_rpaSkin;
};

#endif  // WEEGSKINALIGNMENT_H_
