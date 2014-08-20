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

#include "core/gui/roi/WLROIBox.h"
#include "WLROIControllerSource.h"

const std::string WLROIControllerSource::CLASS = "WLROIControllerSource";

WLROIControllerSource::WLROIControllerSource( osg::ref_ptr< WROI > roi,
                typename WLROIController< WLEMMSurface, std::list< size_t > >::DataTypeSPtr data ) :
                WLROIController( roi, data )
{
}

WLROIControllerSource::~WLROIControllerSource()
{
}

void WLROIControllerSource::recalculate()
{
    if( !m_data )
    {
        return;
    }

    if( !m_dirty )
    {
        return;
    }

    if( !osg::dynamic_pointer_cast< WLROIBox >( m_roi ).get() )
    {
        return;
    }

    osg::ref_ptr< WLROIBox > box = osg::dynamic_pointer_cast< WLROIBox >( m_roi );

    wlog::debug( CLASS ) << "recalculate() WLROIBox";

    m_filter->clear(); // clear the list

    for(size_t i = 0; i < m_data->getVertex()->size(); ++i) // iterate all vertices
    {
        WPosition pos = m_data->getVertex()->at(i);

        if( box->getMaxPos() < pos || pos < box->getMinPos() )
        {
            continue;
        }

        m_filter->push_back(i);
    }

    m_dirty = false;

}
