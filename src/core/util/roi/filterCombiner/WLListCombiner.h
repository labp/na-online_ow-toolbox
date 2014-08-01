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

#ifndef WLLISTCOMBINER_H_
#define WLLISTCOMBINER_H_

#include <list>

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

#include "WLROIFilterCombiner.h"

/**
 *
 */
template< typename T >
class WLListCombiner: public WLROIFilterCombiner
{
public:

    /**
     * Destroys the WLListCombiner.
     */
    virtual ~WLListCombiner();

protected:

    /**
     * The first filter.
     */
    boost::shared_ptr< std::list< T > > m_filter1;

    /**
     * The second filter.
     */
    boost::shared_ptr< std::list< T > > m_filter2;

    /**
     * Interface method to combine two filter structures.
     *
     * @return Returns true if combining was successfull, otherwise false.
     */
    bool combine();

    /**
     * Sets the filters to combine.
     *
     * @param filter1 The first filter.
     * @param filter2 The second filter.
     */
    void setFilterImpl( boost::any const & filter1, boost::any const & filter2 );

    /**
     * Gets the combined filter structure.
     *
     * @return Returns a @boost::any object.
     */
    boost::any getFilterImpl() const;
};

template< typename T >
inline WLListCombiner< T >::~WLListCombiner()
{
}

template< typename T >
inline bool WLListCombiner< T >::combine()
{
    m_filter1->merge( m_filter2.get() );

    return true;
}

template< typename T >
inline void WLListCombiner< T >::setFilterImpl( const boost::any& filter1, const boost::any& filter2 )
{
    m_filter1 = boost::any_cast< boost::shared_ptr< std::list< T > > >( filter1 );
    m_filter2 = boost::any_cast< boost::shared_ptr< std::list< T > > >( filter2 );
}

template< typename T >
inline boost::any WLListCombiner< T >::getFilterImpl() const
{
    return m_filter1;
}

#endif /* WLLISTCOMBINER_H_ */
