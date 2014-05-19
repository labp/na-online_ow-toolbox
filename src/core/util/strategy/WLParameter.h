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

#ifndef WLPARAMETER_H_
#define WLPARAMETER_H_

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

/**
 * WLParameter is the base class for any parameter classes.
 */
class WLParameter: public boost::enable_shared_from_this< WLParameter >
{
public:

    /**
     * A shared pointer on a WLParameter.
     */
    typedef boost::shared_ptr< WLParameter > SPtr;

    /**
     * A shared pointer on a constant WLParameter.
     */
    typedef boost::shared_ptr< const WLParameter > ConstSPtr;

    /**
     * Destroys the WLParameter.
     */
    virtual ~WLParameter();

    /**
     * Gets the parameter name.
     *
     * @return Returns a constant string.
     */
    std::string getName() const;

    /**
     * Sets the parameter name.
     *
     * @param name The name as string.
     */
    void setName( std::string name );

    /**
     * Cast to a concrete parameter type.
     *
     * @return Shared Pointer< Param >
     */
    template< typename Param >
    boost::shared_ptr< Param > getAs()
    {
        return boost::dynamic_pointer_cast< Param >( shared_from_this() );
    }

    /**
     * Cast to a concrete parameter type.
     *
     * @return Shared Pointer< const Param >
     */
    template< typename Param >
    boost::shared_ptr< const Param > getAs() const
    {
        return boost::dynamic_pointer_cast< Param >( shared_from_this() );
    }

protected:

    /**
     * Constructs a new WLParameter.
     *
     * @param name
     */
    WLParameter( const std::string name );

    /**
     * The parameter name.
     */
    std::string m_name;
};

#endif /* WLPARAMETER_H_ */
