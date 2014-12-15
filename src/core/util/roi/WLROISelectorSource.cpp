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

#include "core/gui/roi/WLROIBox.h"
#include "core/util/roi/controllerFactory/WLROICtrlFactorySource.h"
#include "core/util/roi/filterCombiner/WLListCombiner.h"
#include "WLROISelectorSource.h"

const std::string WLROISelectorSource::CLASS = "WLROISelectorSource";

WLROISelectorSource::WLROISelectorSource( WLEMMSurface::SPtr data, WLEMDDrawable3D::SPtr drawable3D ) :
                WLROISelector( data,
                                boost::shared_ptr<
                                                WLROICtrlFactory< WLROIController< WLEMMSurface, std::list< size_t > >,
                                                                WLEMMSurface > >( new WLROICtrlFactorySource ),
                                boost::shared_ptr< WLROIFilterCombiner< std::list< size_t > > >( new WLListCombiner< size_t > ) ), m_drawable3D(
                                drawable3D )
{
}

void WLROISelectorSource::slotAddRoi( osg::ref_ptr< WROI > newRoi )
{
    WLROISelector< WLEMMSurface, std::list< size_t > >::slotAddRoi( newRoi );

    if( !m_drawable3D )
    {
        return;
    }

    m_drawable3D->getWidget()->getScene()->addChild( newRoi.get() );
}

void WLROISelectorSource::slotRemoveRoi( osg::ref_ptr< WROI > ref_ptr )
{
    WLROISelector< WLEMMSurface, std::list< size_t > >::slotRemoveRoi( ref_ptr );

    if( !m_drawable3D )
    {
        return;
    }

    m_drawable3D->getWidget()->getScene()->removeChild( ref_ptr.get() );
}
