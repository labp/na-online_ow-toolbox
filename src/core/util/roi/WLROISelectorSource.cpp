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

#include <boost/foreach.hpp>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"

#include "controllerFactory/WLROICtrlFactorySource.h"
#include "filterCombiner/WLListCombiner.h"
#include "WLROISelectorSource.h"

const std::string WLROISelectorSource::CLASS = "WLROISelectorSource";

WLROISelectorSource::WLROISelectorSource( WLEMData::SPtr data, WLEMDDrawable3D::SPtr drawable3D ) :
                WLROISelector( data ), m_drawable3D( drawable3D )
{
    // create a controller factory first
    m_factory.reset(
                    ( WLROICtrlFactory< WLROIController< WLEMData, std::list< size_t > >, WLEMData, std::list< size_t > >* )new WLROICtrlFactorySource );

    m_combiner.reset( new WLListCombiner< size_t > ); // Init the filter combiner.

    // AFTER init m_factory: create ROIs from the current ROI configuration.
    generateRois(); // involve an existing ROI configuration.
}

/*
 void WLROISelectorSource::recalculate()
 {
 wlog::debug(CLASS) << "recalculate()";

 }
 */

void WLROISelectorSource::slotAddRoi( osg::ref_ptr< WROI > ref_ptr )
{
    WLROISelector< WLEMData, std::list< size_t > >::slotAddRoi( ref_ptr );

    if( !m_drawable3D )
    {
        return;
    }

    m_drawable3D->getWidget()->getScene()->addChild( ref_ptr.get() );
}

void WLROISelectorSource::slotRemoveRoi( osg::ref_ptr< WROI > ref_ptr )
{
    WLROISelector< WLEMData, std::list< size_t > >::slotRemoveRoi( ref_ptr );

    if( !m_drawable3D )
    {
        return;
    }

    m_drawable3D->getWidget()->getScene()->removeChild( ref_ptr.get() );
}
