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

#include <limits>
#include <list>
#include <string>
#include <vector>

#include <Eigen/Dense>  // min/max for WLPositions

#include <core/common/WLogger.h>

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/WLDigPoint.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/enum/WLEPointType.h"
#include "core/data/enum/WLECardinalPoint.h"
#include "core/util/profiler/WLTimeProfiler.h"
#include "WEEGSkinAlignment.h"

const std::string WEEGSkinAlignment::CLASS = "WEEGSkinAlignment";

WEEGSkinAlignment::WEEGSkinAlignment( int maxIterations ) :
                WAlignment( maxIterations )
{
}

WEEGSkinAlignment::~WEEGSkinAlignment()
{
}

void WEEGSkinAlignment::setLpaSkin( const PointT& lpaSkin )
{
    m_lpaSkin = lpaSkin;
}

const WEEGSkinAlignment::PointT& WEEGSkinAlignment::getNasionSkin() const
{
    return m_nasionSkin;
}

void WEEGSkinAlignment::setNasionSkin( const PointT& nasionSkin )
{
    m_nasionSkin = nasionSkin;
}

const WEEGSkinAlignment::PointT& WEEGSkinAlignment::getRpaSkin() const
{
    return m_rpaSkin;
}

void WEEGSkinAlignment::setRpaSkin( const PointT& rpaSkin )
{
    m_rpaSkin = rpaSkin;
}

double WEEGSkinAlignment::align( TransformationT* const matrix, WLEMMeasurement::ConstSPtr emm )
{
    WLTimeProfiler tp( CLASS, __func__ );
    const WLEMMeasurement& emm_ref = *emm;

    // Extract & set corresponding points
    // ----------------------------------
    PointT lpaEEG, nasionEEG, rpaEEG;
    if( extractFiducialPoints( &lpaEEG, &nasionEEG, &rpaEEG, emm_ref ) )
    {
        if( m_lpaSkin != m_nasionSkin && m_nasionSkin != m_rpaSkin )
        {
            const CorrespondenceT corLpa( lpaEEG, m_lpaSkin );
            const CorrespondenceT corNasion( nasionEEG, m_nasionSkin );
            const CorrespondenceT corRpa( rpaEEG, m_rpaSkin );
            addCorrespondence( corLpa );
            addCorrespondence( corNasion );
            addCorrespondence( corRpa );
        }
        else
        {
            wlog::warn( CLASS ) << "Fiducial points for skin are not set!";
        }
    }

    // BEM skin layer: Remove bottom points and transform to base exponent
    // -------------------------------------------------------------------
    PointsT to;
    if( !extractBEMSkinPoints( &to, emm_ref ) )
    {
        return NOT_CONVERGED;
    }

    // Get EEG sensor positions
    // ------------------------
    WLEMDEEG::ConstSPtr eeg;
    if( emm_ref.hasModality( WLEModality::EEG ) )
    {
        eeg = emm_ref.getModality< const WLEMDEEG >( WLEModality::EEG );
    }
    else
    {
        wlog::error( CLASS ) << __func__ << ": No EEG data!";
        return NOT_CONVERGED;
    }
    WLPositions::ConstSPtr fromPtr = eeg->getChannelPositions3d();

    // Compute alignment
    // -----------------
    return WAlignment::align( matrix, *fromPtr, to );
}

bool WEEGSkinAlignment::extractFiducialPoints( PointT* const lpa, PointT* const nasion, PointT* const rpa,
                const WLEMMeasurement& emm )
{
    WLTimeProfiler tp( CLASS, __func__ );
    WLList< WLDigPoint >::SPtr digPoints = emm.getDigPoints( WLEPointType::CARDINAL );
    int count = 0;
    WLList< WLDigPoint >::const_iterator cit;
    for( cit = digPoints->begin(); cit != digPoints->end() && count < 3; ++cit )
    {
        if( cit->checkCardinal( WLECardinalPoint::LPA ) )
        {
            *lpa = cit->getPoint();
            ++count;
        }
        if( cit->checkCardinal( WLECardinalPoint::NASION ) )
        {
            *nasion = cit->getPoint();
            ++count;
        }
        if( cit->checkCardinal( WLECardinalPoint::RPA ) )
        {
            *rpa = cit->getPoint();
            ++count;
        }
    }
    if( count > 2 )
    {
        return true;
    }
    else
    {
        wlog::warn( CLASS ) << "Could not found fiducial points: " << count;
        return false;
    }
}

bool WEEGSkinAlignment::extractBEMSkinPoints( PointsT* const out, const WLEMMeasurement& emm )
{
    WLTimeProfiler tp( CLASS, __func__ );
    WLEMMSubject::ConstSPtr subject = emm.getSubject();
    const std::list< WLEMMBemBoundary::SPtr >& bems = *subject->getBemBoundaries();
    std::list< WLEMMBemBoundary::SPtr >::const_iterator itBem;
    WLEMMBemBoundary::ConstSPtr bemSkin;
    for( itBem = bems.begin(); itBem != bems.end(); ++itBem )
    {
        if( ( *itBem )->getBemType() == WLEBemType::OUTER_SKIN || ( *itBem )->getBemType() == WLEBemType::HEAD )
        {
            bemSkin = *itBem;
            break;
        }
    }
    if( !bemSkin )
    {
        wlog::error( CLASS ) << __func__ << ": No BEM skin layer available!";
        return false;
    }

    const WLPositions& bemPosition = *bemSkin->getVertex();
    const WLPositions::ScalarT min = bemPosition.data().row(2).minCoeff();
    const WLPositions::ScalarT max = bemPosition.data().row(2).maxCoeff();
    const WLPositions::ScalarT z_threashold = min + ( max - min ) * 0.25;
    wlog::debug( CLASS ) << "icpAlign: BEM z_threashold: " << z_threashold;

    PointsT::IndexT idx = 0;
    for( WLPositions::IndexT i = 0; i < bemPosition.size(); ++i )
    {
        if( bemPosition.at( i ).z() > z_threashold )
        {
            ++idx;
        }
    }

    const double factor = WLEExponent::factor( bemPosition.exponent() );
    out->exponent( WLEExponent::BASE );
    PointsT::PositionsT& outPos = out->data();
    outPos.resize( PointsT::PositionsT::RowsAtCompileTime, idx );
    idx = 0;
    for( WLPositions::IndexT i = 0; i < bemPosition.size(); ++i )
    {
        if( bemPosition.at( i ).z() > z_threashold )
        {
            const PointsT::PositionT tmp( bemPosition.at( i ).x(), bemPosition.at( i ).y(), bemPosition.at( i ).z() );
            outPos.col( idx ) = tmp * factor;
            ++idx;
        }
    }

    out->coordSystem( bemPosition.coordSystem() );
    out->unit( bemPosition.unit() );
    out->exponent( WLEExponent::BASE );

    return true;
}
