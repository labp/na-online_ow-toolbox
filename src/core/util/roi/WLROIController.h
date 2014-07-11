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

#ifndef WLROICONTROLLER_H_
#define WLROICONTROLLER_H_

#include <boost/shared_ptr.hpp>

#include <core/graphicsEngine/WROI.h>

/**
 *
 */
template< typename DataType, typename FilterType >
class WLROIController
{
public:

    /**
     * A shared pointer on an instance of the used source data type.
     */
    typedef boost::shared_ptr< DataType > DataTypeSPtr;

    /**
     * A shared pointer on an instance of the used filter structure.
     */
    typedef boost::shared_ptr< FilterType > FilterTypeSPtr;

    WLROIController( osg::ref_ptr< WROI > roi, DataTypeSPtr data );

    /**
     * Destroys the WLROIController.
     */
    virtual ~WLROIController();

    /**
     * Forces the concrete controller to recalculate the filter structure.
     */
    virtual void recalculate() = 0;

    /**
     * Gets the filter structure.
     *
     * @return Returns a shared pointer on the filter structure.
     */
    FilterTypeSPtr getFilter() const;

protected:

    /**
     * The roi object.
     */
    osg::ref_ptr< WROI > m_roi;

    /**
     * The filter data structure.
     */
    FilterTypeSPtr m_filter;
};

#endif /* WLROICONTROLLER_H_ */
