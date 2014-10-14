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

#include <string>

#include "core/data/emd/WLEMDMEG.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLABoundCalculator.h"

const std::string WLABoundCalculator::CLASS = "WLABoundCalculator";

WLABoundCalculator::~WLABoundCalculator()
{
}

WLABoundCalculator::MinMax WLABoundCalculator::getBounds2D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality )
{
    MinMax max( 0.0, 0.0 );

    if( modality == WLEModality::SOURCE && emm->hasModality( modality ) )
    {
        WLEModality::Enum origin_modality = emm->getModality< const WLEMDSource >( modality )->getOriginModalityType();
        max.second = getMax( emm->getModality( origin_modality )->getData() );
        return max;
    }
    if( WLEModality::isMEGCoil( modality ) )
    {
        modality = WLEModality::MEG;
    }
    if( emm->hasModality( modality ) )
    {
        max.second = getMax( emm->getModality( modality )->getData() );
        return max;
    }
    else
    {
        return max;
    }
}

WLABoundCalculator::MinMax WLABoundCalculator::getBounds3D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality )
{
    MinMax bounds( 0.0, 0.0 );

    if( WLEModality::isMEGCoil( modality ) && emm->hasModality( WLEModality::MEG ) )
    {
        WLEMDMEG::ConstSPtr meg = emm->getModality< const WLEMDMEG >( WLEModality::MEG );
        WLEMDMEG::SPtr megCoil;
        if( WLEMDMEG::extractCoilModality( megCoil, meg, modality, true ) )
        {
            bounds.first = getMin( megCoil->getData() );
            bounds.second = getMax( megCoil->getData() );
            return bounds;
        }
    }
    if( emm->hasModality( modality ) )
    {
        bounds.first = getMin( emm->getModality( modality )->getData() );
        bounds.second = getMax( emm->getModality( modality )->getData() );
        return bounds;
    }
    return bounds;
}
