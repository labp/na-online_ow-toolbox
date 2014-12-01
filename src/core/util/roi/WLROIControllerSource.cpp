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

WLROIControllerSource::WLROIControllerSource( osg::ref_ptr< WROI > roi, WLEMMSurface::SPtr data ) :
                WLROIController( roi, data )
{
}

WLROIControllerSource::~WLROIControllerSource()
{
}

void WLROIControllerSource::recalculate()
{
    m_filter.reset( new std::list< size_t > );

    if( !m_dirty )
    {
        return;
    }

    if( !m_data )
    {
        return;
    }

    if( !osg::dynamic_pointer_cast< WLROIBox >( m_roi ).get() )
    {
        return;
    }

    osg::ref_ptr< WLROIBox > box = osg::dynamic_pointer_cast< WLROIBox >( m_roi );

    //wlog::debug( CLASS ) << "recalculate() WLROIBox";
    //wlog::debug( CLASS ) << "Box dimensions: min: " << box->getMinPos() << " | max: " << box->getMaxPos();

    for( size_t i = 0; i < m_data->getVertex()->size(); ++i ) // iterate all vertices
    {
        WPosition pos = m_data->getVertex()->at( i );

        if( box->getMaxPos().x() < pos.x() || pos.x() < box->getMinPos().x() )
        {
            continue;
        }

        if( box->getMaxPos().y() < pos.y() || pos.y() < box->getMinPos().y() )
        {
            continue;
        }

        if( box->getMaxPos().z() < pos.z() || pos.z() < box->getMinPos().z() )
        {
            continue;
        }

        m_filter->push_back( i );
    }

    wlog::debug( CLASS ) << "Vertices found: " << m_filter->size();

    /*
    if( m_filter->size() > 0 )
    {
        int i = 0;
        for( std::list< size_t >::iterator it = m_filter->begin(); i < 10 && it != m_filter->end(); ++it )
        {
            wlog::debug( CLASS ) << "Source " << *it << " Point: " << m_data->getVertex()->at( *it );
            ++i;
        }
    }
    */

    m_dirty = false;

}
