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

#ifndef WLROICTRLCREATOR_H_
#define WLROICTRLCREATOR_H_

#include <boost/shared_ptr.hpp>

#include <core/graphicsEngine/WROI.h>

/**
 * The WLROICtrlCreator provides an interface to create an instance of @Base under the
 * condition of @DataType and @FilterType.
 */
template< typename Base, typename DataType = void >
class WLROICtrlCreator
{
public:

    /**
     * A shared pointer on a WLROICtrlCreator.
     */
    typedef boost::shared_ptr< WLROICtrlCreator< Base, DataType > > type;

    /**
     * Destroys the WLROICtrlCreator.
     */
    virtual ~WLROICtrlCreator();

    /**
     * Interface method for creating a new @Base object by the derived class with the defined
     * parameter @roi and @data.
     *
     * @param roi The ROI.
     * @param data The data container.
     * @return
     */
    virtual Base *create( osg::ref_ptr< WROI > roi, boost::shared_ptr< DataType > data ) = 0;
};

template< typename Base, typename DataType >
inline WLROICtrlCreator< Base, DataType >::~WLROICtrlCreator()
{

}

#endif /* WLROICTRLCREATOR_H_ */
