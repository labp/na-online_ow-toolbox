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

#ifndef WLROICONTROLLERSOURCE_H_
#define WLROICONTROLLERSOURCE_H_

#include <list>

#include <core/graphicsEngine/WROI.h>

#include "core/data/emd/WLEMData.h"
#include "WLROIController.h"

/**
 * The WLROIControllerSource class represents a ROI controller for selecting sources while the
 * source reconstruction algorithm.
 * The class uses the following structures:
 *      DataType: WLEMData - data structure and base for the source reconstruction
 *      FilterType: list<Indices> - list, selecting the calculated channel indices.
 */
class WLROIControllerSource: public WLROIController< WLEMData, std::list< size_t > >
{

public:

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WLROIControllerSource.
     *
     * @param roi The region of interest.
     * @param data The data container.
     */
    WLROIControllerSource( osg::ref_ptr< WROI > roi, WLEMData::SPtr data );

    /**
     * Destroys the WLROIControllerSource.
     */
    virtual ~WLROIControllerSource();

protected:

    /**
     * Recalculates the filter structure depending on the current ROI configuration.
     */
    virtual void recalculate();
};

#endif /* WLROICONTROLLERSOURCE_H_ */
