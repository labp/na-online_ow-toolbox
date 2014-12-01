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

#ifndef WLARRAYLIST_H_
#define WLARRAYLIST_H_

#include <ostream>
#include <vector>

#include <boost/shared_ptr.hpp>

/**
 * Helper class for std::vector. Main purpose is to provide an easy-to-use Smart Pointer access.\n
 * ATTENTION: Due to a non-virtual destructor of std::vector and memory leak prevention, member variables are not allowed!\n
 * \n
 * Correct:
 * \code
 * WLArrayList::SPtr v1 = WLArrayList::instance();
 * WLArrayList::SPtr v2( new WLArrayList() );
 * WLArrayList v3();
 * \endcode
 *
 * Wrong:
 * \code
 * std::vector* v = new WLArrayList();
 * delete v; // Does not call destructor for WLArrayList!
 * \endcode
 *
 * \author pieloth
 * \ingroup container
 */
template< typename T >
class WLArrayList: public std::vector< T >
{
public:
    typedef boost::shared_ptr< WLArrayList< T > > SPtr;

    typedef boost::shared_ptr< const WLArrayList< T > > ConstSPtr;

    explicit WLArrayList();

    explicit WLArrayList( const std::vector< T >& x );

    virtual ~WLArrayList();

    static WLArrayList< T >::SPtr instance();

    static WLArrayList< T >::SPtr instance( const std::vector< T >& x );
};

template< typename T >
WLArrayList< T >::WLArrayList() :
                std::vector< T >()
{
}

template< typename T >
WLArrayList< T >::WLArrayList( const std::vector< T >& x ) :
                std::vector< T >( x )
{
}

template< typename T >
WLArrayList< T >::~WLArrayList()
{
}

template< typename T >
inline typename WLArrayList< T >::SPtr WLArrayList< T >::instance()
{
    return WLArrayList< T >::SPtr( new WLArrayList< T > );
}

template< typename T >
inline typename WLArrayList< T >::SPtr WLArrayList< T >::instance( const std::vector< T >& x )
{
    return WLArrayList< T >::SPtr( new WLArrayList< T >( x ) );
}

template< typename T >
std::ostream& operator<<( std::ostream &strm, const WLArrayList< T >& obj )
{
    strm << "WLArrayList.size=" << obj.size();
#ifdef DEBUG
    if( !obj.empty() )
    {
        strm << "\n";
        strm << "[";
        typename WLArrayList< T >::const_iterator it;
        for( it = obj.begin(); it != obj.end(); ++it )
        {
            strm << *it << ", ";
        }
        strm << "]";
    }
#endif
    return strm;
}

#endif  // WLARRAYLIST_H_
