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

#include <core/common/WLogger.h>
#include <core/graphicsEngine/WROIBox.h>

#include "WLROIControllerSource.h"

const std::string WLROIControllerSource::CLASS = "WLROIControllerSource";

WLROIControllerSource::WLROIControllerSource( osg::ref_ptr< WROI > roi,
                typename WLROIController< WLEMData, std::list< size_t > >::DataTypeSPtr data ) :
                WLROIController( roi, data )
{
}

WLROIControllerSource::~WLROIControllerSource()
{
}

void WLROIControllerSource::recalculate()
{
    if( osg::dynamic_pointer_cast< WROIBox >( m_roi ).get() )
    {
        wlog::debug( CLASS ) << "recalculate() WROIBox";
    }
    else
    {
        wlog::debug( CLASS ) << "recalculate()";
    }
}
