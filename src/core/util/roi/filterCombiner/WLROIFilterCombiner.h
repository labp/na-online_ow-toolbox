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

#ifndef WLROIFILTERCOMBINER_H_
#define WLROIFILTERCOMBINER_H_

#include <boost/any.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

/**
 * The WLROIFilterCombiner provides an interface to combine the ROI-FilterType structures.
 * Derived classes have to implement the combining method for a concrete type.
 */
class WLROIFilterCombiner: public boost::enable_shared_from_this< WLROIFilterCombiner >
{
public:

    /**
     * A shared pointer on a WLROIFilterCombiner.
     */
    typedef boost::shared_ptr< WLROIFilterCombiner > SPtr;

    /**
     * Destroys the WLROIFilterCombiner.
     */
    virtual ~WLROIFilterCombiner();

    /**
     * Casts the WLROIFilterCombiner to type T.
     *
     * @return Returns a shared pointer on a T.
     */
    template< typename T >
    boost::shared_ptr< T > getAs()
    {
        return boost::dynamic_pointer_cast< T >( shared_from_this() );
    }

    /**
     * Casts the WLROIFilterCombiner to type T.
     *
     * @return Returns a shared pointer on a constant T.
     */
    template< typename T >
    boost::shared_ptr< const T > getAs() const
    {
        return boost::dynamic_pointer_cast< T >( shared_from_this() );
    }

    /**
     * Sets the filters to combine.
     *
     * @param filter1 The first filter.
     * @param filter2 The second filter.
     */
    template< typename FilterType >
    void setFilter( boost::shared_ptr< FilterType > filter1, boost::shared_ptr< FilterType > filter2 );

    /**
     * Gets the combined filter structure.
     *
     * @return Returns a @FilterType object.
     */
    template< typename FilterType >
    boost::shared_ptr< FilterType > getFilter();

protected:

    /**
     * Interface method to combine two filter structures.
     *
     * @return Returns true if combining was successfull, otherwise false.
     */
    virtual bool combine() = 0;

    /**
     * Sets the filters to combine.
     *
     * @param filter1 The first filter.
     * @param filter2 The second filter.
     */
    virtual void setFilterImpl( boost::any const & filter1, boost::any const & filter2 ) = 0;

    /**
     * Gets the combined filter structure.
     *
     * @return Returns a @boost::any object.
     */
    virtual boost::any getFilterImpl() const = 0;
};

template< typename FilterType >
inline void WLROIFilterCombiner::setFilter( boost::shared_ptr< FilterType > filter1, boost::shared_ptr< FilterType > filter2 )
{
    setFilterImpl( boost::any( filter1 ), boost::any( filter2 ) );
}

template< typename FilterType >
inline boost::shared_ptr< FilterType > WLROIFilterCombiner::getFilter()
{
    if( combine() )
    {
        boost::any res = getFilterImpl();
        return boost::any_cast< boost::shared_ptr< FilterType > >( res );
    }

    return boost::any();
}

#endif /* WLROIFILTERCOMBINER_H_ */
