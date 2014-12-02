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

#ifndef WLROICONTROLLERSOURCE_H_
#define WLROICONTROLLERSOURCE_H_

#include <list>

#include <core/graphicsEngine/WROI.h>

#include "core/data/WLEMMSurface.h"
#include "WLROIController.h"

/**
 * The WLROIControllerSource class represents a ROI controller for selecting sources while the
 * source reconstruction algorithm.\n
 * The class uses the following structures:\n
 *      DataType: WLEMMSurface - the surface data containing the 3D head models vertices.\n
 *      FilterType: list<Indices> - list, selecting the calculated channel indices.\n
 *
 * \author maschke
 * \ingroup util
 */
class WLROIControllerSource: public WLROIController< WLEMMSurface, std::list< size_t > >
{
public:
    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructs a new WLROIControllerSource.
     *
     * \param roi The region of interest.
     * \param data The data container.
     */
    WLROIControllerSource( osg::ref_ptr< WROI > roi, WLEMMSurface::SPtr data );

    /**
     * Destroys the WLROIControllerSource.
     */
    virtual ~WLROIControllerSource();

    /**
     * Recalculates the filter structure depending on the current ROI configuration.
     */
    virtual void recalculate();
};

#endif  // WLROICONTROLLERSOURCE_H_
