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

#ifndef WLSET_H_
#define WLSET_H_

#include <ostream>
#include <set>

#include <boost/shared_ptr.hpp>

/**
 * Helper class for std::set. Main purpose is to provide an easy-to-use Smart Pointer access.\n
 * ATTENTION: Due to a non-virtual destructor of std::set and memory leak prevention, member variables are not allowed!\n
 * \n
 * Correct:
 * \code
 * WLSet::SPtr s1 = WLSet::instance();
 * WLSet::SPtr s2( new WLSet() );
 * WLSet s3();
 * \endcode
 *
 * Wrong:
 * \code
 * std::set* s = new WLSet();
 * delete s; // Does not call destructor for WLArrayList!
 * \endcode
 *
 * \author pieloth
 * \ingroup container
 */
template< typename T >
class WLSet: public std::set< T >
{
public:
    typedef boost::shared_ptr< WLSet< T > > SPtr;

    typedef boost::shared_ptr< const WLSet< T > > ConstSPtr;

    explicit WLSet();

    explicit WLSet( const std::set< T >& x );

    virtual ~WLSet();

    static WLSet< T >::SPtr instance();

    static WLSet< T >::SPtr instance( const std::set< T >& x );
};

template< typename T >
WLSet< T >::WLSet( const std::set< T >& x ) :
                std::set< T >( x )
{
}

template< typename T >
WLSet< T >::~WLSet()
{
}

template< typename T >
inline typename WLSet< T >::SPtr WLSet< T >::instance()
{
    return WLSet< T >::SPtr( new WLSet< T > );
}

template< typename T >
inline typename WLSet< T >::SPtr WLSet< T >::instance( const std::set< T >& x )
{
    return WLSet< T >::SPtr( new WLSet< T >( x ) );
}

template< typename T >
std::ostream& operator<<( std::ostream &strm, const WLSet< T >& obj )
{
    strm << "WLSet.size=" << obj.size();
#ifdef DEBUG
    if( !obj.empty() )
    {
        strm << "\n";
        strm << "[";
        typename WLSet< T >::const_iterator it;
        for( it = obj.begin(); it != obj.end(); ++it )
        {
            strm << *it << ", ";
        }
        strm << "]";
    }
#endif
    return strm;
}

#endif  // WLSET_H_
