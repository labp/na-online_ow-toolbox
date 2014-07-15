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

#ifndef WLROICTRLFACTORYSOURCE_H_
#define WLROICTRLFACTORYSOURCE_H_

#include <vector>

#include "core/data/emd/WLEMData.h"
#include "WLROIControllerSource.h"
#include "WLROICtrlFactory.h"

/**
 * WLROICtrlFactorySource is a implementation of the abstract factory WLROICtrlFactory to create new
 * instances of the WLROIControllerSource class.
 */
class WLROICtrlFactorySource: public WLROICtrlFactory< WLROIControllerSource, WLEMData, std::vector< size_t > >
{
public:

    /**
     * Creates a new instance of a WLROIControllerSource.
     *
     * @param name The instance name.
     * @param roi The ROI.
     * @param data The data container.
     * @return Returns a pointer on the new WLROIControllerSource instance.
     */
    WLROIControllerSource *create( const std::string& name, osg::ref_ptr< WROI > roi, boost::shared_ptr< WLEMData > data ) const;
};

#endif /* WLROICTRLFACTORYSOURCE_H_ */
