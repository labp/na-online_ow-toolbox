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

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/data/enum/WLEModality.h"

#include "WLEMDDrawable2DSingleChannel.h"
#include "WLEMDDrawable2DSingleSource.h"

const std::string WLEMDDrawable2DSingleSource::CLASS = "WLEMDDrawable2DSingleSource";

WLEMDDrawable2DSingleSource::WLEMDDrawable2DSingleSource( WUIViewWidget::SPtr widget ) :
                WLEMDDrawable2DSingleChannel( widget )
{
}

WLEMDDrawable2DSingleSource::~WLEMDDrawable2DSingleSource()
{
}

void WLEMDDrawable2DSingleSource::draw( WLEMMeasurement::SPtr emm )
{
    bool success = false;
    if( emm->hasModality( WLEModality::SOURCE ) )
    {
        WLEMDSource::ConstSPtr emd = emm->getModality< WLEMDSource >( WLEModality::SOURCE );
        if( emd )
        {
            setModality( emd->getOriginModalityType() );
            WLEMDDrawable2DSingleChannel::draw( emm );
            success = true;
        }
    }
    if( !success )
    {
        wlog::error( CLASS ) << "Could not retrieve origin modality!";
    }
}
