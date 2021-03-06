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

#ifndef WLROICTRLFACTORYSOURCE_H_
#define WLROICTRLFACTORYSOURCE_H_

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMSurface.h"
#include "core/util/roi/WLROIControllerSource.h"
#include "WLROICtrlFactory.h"

/**
 * WLROICtrlFactorySource is a implementation of the abstract factory WLROICtrlFactory to create new
 * instances of the WLROIControllerSource class.
 *
 * \author maschke
 * \ingroup util
 */
class WLROICtrlFactorySource: public WLROICtrlFactory< WLROIController< WLEMMSurface, std::list< size_t > >, WLEMMSurface >
{
public:
    /**
     * A shared pointer on a WLROICtrlFactorySource.
     */
    typedef boost::shared_ptr< WLROICtrlFactorySource > SPtr;

    /**
     * Contructs an new WLROICtrlFactorySource.
     */
    WLROICtrlFactorySource();

    /**
     * Creates a new instance of a WLROIControllerSource.
     *
     * \param name The instance name.
     * \param roi The ROI.
     * \param data The data container.
     * \return Returns a pointer on the new WLROIControllerSource instance.
     */
    WLROIController< WLEMMSurface, std::list< size_t > > *create( const std::string& name, osg::ref_ptr< WROI > roi,
                    boost::shared_ptr< WLEMMSurface > data ) const;
};

#endif  // WLROICTRLFACTORYSOURCE_H_
