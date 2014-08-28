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

#include <boost/shared_ptr.hpp>

#include "WLROIFilterCombiner.h"

/**
 *
 */
template< typename T >
class WLListCombiner: public WLROIFilterCombiner< std::list< T > >
{
public:

    /**
     * Destroys the WLListCombiner.
     */
    virtual ~WLListCombiner();

    /**
     * Interface method to combine two filter structures.
     *
     * @return Returns true if combining was successful, otherwise false.
     */
    virtual bool combine();

};

template< typename T >
inline WLListCombiner< T >::~WLListCombiner()
{
}

template< typename T >
inline bool WLListCombiner< T >::combine()
{
    std::list< T >& list = *( WLROIFilterCombiner< std::list< T > >::m_first );
    list.merge( *( WLROIFilterCombiner< std::list< T > >::m_second ) );
    list.sort();
    list.unique();

    return true;
}

#endif /* WLLISTCOMBINER_H_ */
