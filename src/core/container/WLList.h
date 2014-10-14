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

#ifndef WLLIST_H_
#define WLLIST_H_

#include <list>
#include <ostream>

#include <boost/shared_ptr.hpp>

/**
 * Helper class for std::list. Main purpose is to provide an easy-to-use Smart Pointer access.
 * ATTENTION: Due to a non-virtual destructor of std::list and memory leak prevention, member variables are not allowed!
 *
 * Correct:
 * WLList::SPtr l1 = WLList::instance();
 * WLList::SPtr l2( new WLList() );
 * WLList l3();
 *
 * Wrong:
 * std::list* l = new WLList();
 * delete l; // Does not call destructor for WLArrayList!
 *
 * \author pieloth
 */
template< typename T >
class WLList: public std::list< T >
{
public:
    typedef boost::shared_ptr< WLList< T > > SPtr;

    typedef boost::shared_ptr< const WLList< T > > ConstSPtr;

    explicit WLList();

    explicit WLList( const std::list< T >& x );

    virtual ~WLList();

    static WLList< T >::SPtr instance();

    static WLList< T >::SPtr instance( const std::list< T >& x );
};

template< typename T >
WLList< T >::WLList() :
                std::list< T >()
{
}

template< typename T >
WLList< T >::WLList( const std::list< T >& x ) :
                std::list< T >( x )
{
}

template< typename T >
WLList< T >::~WLList()
{
}

template< typename T >
inline typename WLList< T >::SPtr WLList< T >::instance()
{
    return WLList< T >::SPtr( new WLList< T > );
}

template< typename T >
inline typename WLList< T >::SPtr WLList< T >::instance( const std::list< T >& x )
{
    return WLList< T >::SPtr( new WLList< T >( x ) );
}

template< typename T >
std::ostream& operator<<( std::ostream &strm, const WLList< T >& obj )
{
    strm << "WLList.size=" << obj.size();
#ifdef DEBUG
    if( !obj.empty() )
    {
        strm << "\n";
        strm << "[";
        typename WLList< T >::const_iterator it;
        for( it = obj.begin(); it != obj.end(); ++it )
        {
            strm << *it << ", ";
        }
        strm << "]";
    }
#endif
    return strm;
}

#endif  // WLLIST_H_
