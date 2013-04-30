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

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/gui/WCustomWidget.h>

#include "core/data/WLDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"
#include "core/data/emd/WLEMDSource.h"

#include "WLEMDDrawable2DSingleChannel.h"
#include "WLEMDDrawable2DSingleSource.h"

namespace LaBP
{
    const std::string WLEMDDrawable2DSingleSource::CLASS = "WLEMDDrawable2DSingleSource";

    WLEMDDrawable2DSingleSource::WLEMDDrawable2DSingleSource( WCustomWidget::SPtr widget ) :
                    WLEMDDrawable2DSingleChannel( widget )
    {
    }

    WLEMDDrawable2DSingleSource::~WLEMDDrawable2DSingleSource()
    {
    }

    void WLEMDDrawable2DSingleSource::draw( LaBP::WLDataSetEMM::SPtr emm )
    {
        bool success = false;
        if( emm->hasModality( WEModalityType::SOURCE ) )
        {
            WLEMDSource::ConstSPtr emd = emm->getModality< WLEMDSource >( WEModalityType::SOURCE );
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

} /* namespace LaBP */
