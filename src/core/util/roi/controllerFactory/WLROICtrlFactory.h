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

#ifndef WLROICTRLFACTORY_H_
#define WLROICTRLFACTORY_H_
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "WLROICtrlCreator.h"
#include "WLROICtrlFactoryBase.h"

/**
 * WLROICtrlFactory defines an interface for constructing new instances of @Base by derived
 * factories.
 */
template< typename Base, typename DataType = void, typename FilterType = void >
class WLROICtrlFactory: public WLROICtrlFactoryBase< typename WLROICtrlCreator< Base, DataType, FilterType >::type >,
                public boost::enable_shared_from_this< WLROICtrlFactory< Base, DataType, FilterType > >
{
public:

    /**
     * A shared pointer on a WLROICtrlFactory.
     */
    typedef boost::shared_ptr< WLROICtrlFactory > SPtr;

    /**
     * Destroys the WLROICtrlFactory.
     */
    virtual ~WLROICtrlFactory();

    /**
     * Interface method for creating a new @Base instance.
     *
     * @param name The instance name.
     * @param roi The ROI.
     * @param data The data container.
     * @return Returns a pointer on the new @Base instance.
     */
    virtual Base* create( const std::string& name, osg::ref_ptr< WROI > roi, boost::shared_ptr< DataType > data ) const = 0;

    template< typename T >
    boost::shared_ptr< T > getAs()
    {
        return boost::dynamic_pointer_cast< T >( this->shared_from_this() );
    }

    template< typename T >
    boost::shared_ptr< const T > getAs() const
    {
        return boost::dynamic_pointer_cast< T >( this->shared_from_this() );
    }
};

template< typename Base, typename DataType, typename FilterType >
inline WLROICtrlFactory< Base, DataType, FilterType >::~WLROICtrlFactory()
{

}

#endif /* WLROICTRLFACTORY_H_ */
