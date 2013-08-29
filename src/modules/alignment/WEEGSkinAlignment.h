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

#ifndef WEEGSKINALIGNMENT_H_
#define WEEGSKINALIGNMENT_H_

#include <string>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLEMMeasurement.h"

#include "WAlignment.h"

/**
 * TODO(pieloth): explain automatic matching with fiducial points
 */
class WEEGSkinAlignment: public WAlignment
{
public:
    static const std::string CLASS;

    WEEGSkinAlignment( int maxInteration = 10 );
    virtual ~WEEGSkinAlignment();

    double align( TransformationT* const matrix, WLEMMeasurement::ConstSPtr emm );

    const WPosition& getLpaSkin() const;

    void setLpaSkin( const WPosition& lpaSkin );

    const WPosition& getNasionSkin() const;

    void setNasionSkin( const WPosition& nasionSkin );

    const WPosition& getRpaSkin() const;

    void setRpaSkin( const WPosition& rpaSkin );

private:
    bool extractFiducialPoints( WPosition* const lpa, WPosition* const nasion, WPosition* const rpa, const WLEMMeasurement& emm );

    bool extractBEMSkinPoints( PointsT* const out, const WLEMMeasurement& emm);

    WPosition m_lpaSkin;
    WPosition m_nasionSkin;
    WPosition m_rpaSkin;
};

#endif  // WEEGSKINALIGNMENT_H_
