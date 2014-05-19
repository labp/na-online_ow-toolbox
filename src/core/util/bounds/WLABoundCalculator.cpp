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

#include <boost/assign.hpp>
#include <boost/foreach.hpp>

#include <Eigen/Core>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLABoundCalculator.h"

using namespace boost::assign;

const std::string WLABoundCalculator::CLASS = "WLABoundCalculator";

WLABoundCalculator::~WLABoundCalculator()
{
}

WLArrayList< WLEMData::ScalarT > WLABoundCalculator::getBounds2D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality )
{
    WLArrayList< WLEMData::ScalarT > max;

    if( modality == WLEModality::SOURCE && emm->hasModality( modality ) )
    {
        WLEModality::Enum origin_modality = emm->getModality< const WLEMDSource >( modality )->getOriginModalityType();

        max += getMax( emm->getModality( origin_modality )->getData() );

        return max;
    }
    if( WLEModality::isMEGCoil( modality ) )
    {
        modality = WLEModality::MEG;
    }
    if( emm->hasModality( modality ) )
    {
        max += getMax( emm->getModality( modality )->getData() );

        return max;
    }
    else
    {
        max += 0;

        return max;
    }
}

WLArrayList< WLEMData::ScalarT > WLABoundCalculator::getBounds3D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality )
{
    WLArrayList< WLEMData::ScalarT > bounds;

    if( emm->hasModality( modality ) )
    {
        bounds += getMax( emm->getModality( modality )->getData() );
        bounds += getMin( emm->getModality( modality )->getData() );

        return bounds;
    }
    if( WLEModality::isMEGCoil( modality ) && emm->hasModality( WLEModality::MEG ) )
    {
        WLEMDMEG::ConstSPtr meg = emm->getModality< const WLEMDMEG >( WLEModality::MEG );
        WLEMDMEG::SPtr megCoil;
        if( WLEMDMEG::extractCoilModality( megCoil, meg, modality, true ) )
        {
            bounds += getMax( megCoil->getData() );
            bounds += getMin( megCoil->getData() );
            return bounds;
        }
        else
        {
            bounds += 0, 0;
            return bounds;
        }
    }
    else
    {
        bounds += 0, 0;
        return bounds;
    }
}
