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

#ifndef WLGENERICPARAMETER_H_
#define WLGENERICPARAMETER_H_

#include "WLParameter.h"

/**
 * WLGenericParameter extends the WLParameter class by the appearance of a generic value container.
 */
template< typename T >
class WLGenericParameter: public WLParameter
{
public:

    /**
     * A shared pointer on a WLGenericParameter.
     */
    typedef boost::shared_ptr< WLGenericParameter< T > > SPtr;

    /**
     * A shared pointer on a constant WLGenericParameter.
     */
    typedef boost::shared_ptr< const WLGenericParameter< T > > ConstSPtr;

    /**
     * Constructs a new WLGenericParameter.
     */
    WLGenericParameter( const T& value );

    /**
     * Constructs a new WLGenericParameter with a defined value.
     *
     * @param value The default parameter value.
     */
    WLGenericParameter( std::string name, const T& value );

    /**
     * Destroys the WLGenericParameter.
     */
    virtual ~WLGenericParameter();

    /**
     * Gets the value.
     *
     * @return Returns the value type T.
     */
    T& getValue();

    /**
     * Sets the value.
     *
     * @param value The value.
     */
    void setValue( const T& value );

protected:

    /**
     * The parameter value.
     */
    T m_value;
};

template< typename T >
inline WLGenericParameter< T >::WLGenericParameter( const T& value ) :
                WLParameter( "" ), m_value( value )
{
}

template< typename T >
inline WLGenericParameter< T >::WLGenericParameter( std::string name, const T& value ) :
                WLParameter( name ), m_value( value )
{
}

template< typename T >
inline WLGenericParameter< T >::~WLGenericParameter()
{
}

template< typename T >
inline T& WLGenericParameter< T >::getValue()
{
    return m_value;
}

template< typename T >
inline void WLGenericParameter< T >::setValue( const T& value )
{
    m_value = value;
}

/**
 * Represents a double parameter.
 */
typedef WLGenericParameter< double > WLParamDouble;

/**
 * Represents a integer parameter.
 */
typedef WLGenericParameter< int > WLParamInt;

#endif /* WLGENERICPARAMETER_H_ */
