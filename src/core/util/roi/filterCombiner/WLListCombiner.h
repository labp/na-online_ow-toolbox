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

#ifndef WLLISTCOMBINER_H_
#define WLLISTCOMBINER_H_

#include <list>

#include <boost/shared_ptr.hpp>

#include "WLROIFilterCombiner.h"

/**
 * \author maschke
 * \ingroup util
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
     * \return Returns true if combining was successful, otherwise false.
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

#endif  // WLLISTCOMBINER_H_
