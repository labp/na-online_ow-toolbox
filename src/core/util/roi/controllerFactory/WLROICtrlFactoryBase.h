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

#ifndef WLROICTRLFACTORYBASE_H_
#define WLROICTRLFACTORYBASE_H_

#include <map>
#include <string>

/**
 *
 */
template< typename Creator >
class WLROICtrlFactoryBase
{
protected:

    /**
     * Abbreviation for the creator directory.
     */
    typedef std::map< std::string, Creator > tableCreator;

    /**
     * An constant iterator on the creator directory.
     */
    typedef typename tableCreator::const_iterator citerTc;

    /**
     * Hides the standard constructor.
     */
    WLROICtrlFactoryBase();

    /**
     * Hides the copy constructor.
     *
     * @param The reference to copy.
     */
    WLROICtrlFactoryBase( const WLROICtrlFactoryBase& );

    /**
     * Hides the assign operator.
     *
     * @param The reference to assign.
     * @return Returns a reference on a WLROICtrlFactoryBase
     */
    WLROICtrlFactoryBase& operator=( const WLROICtrlFactoryBase& );

    /**
     * Method to register a controller creator at the factory.
     *
     * @param name The unique name of the creator.
     * @param creator The creator instance.
     */
    void registerCreator( const std::string& name, Creator creator );

    /**
     * Gets the begin iterator for the creators.
     *
     * @return Returns a constant iterator.
     */
    citerTc begin() const;

    /**
     * Gets the end iterator for the creators.
     *
     * @return Returns a constant iterator.
     */
    citerTc end() const;

    /**
     * Method to find a creator for the @name.
     *
     * @param name The name to look for.
     * @return Returns a constant iterator.
     */
    citerTc find( const std::string& name ) const;

    /**
     * The creator directory.
     */
    tableCreator m_creators;
};

template< typename Creator >
inline WLROICtrlFactoryBase< Creator >::WLROICtrlFactoryBase()
{
}

template< typename Creator >
inline WLROICtrlFactoryBase< Creator >::WLROICtrlFactoryBase( const WLROICtrlFactoryBase& constWLROICtrlFactoryBase )
{
}

template< typename Creator >
inline WLROICtrlFactoryBase< Creator >& WLROICtrlFactoryBase< Creator >::operator=( const WLROICtrlFactoryBase< Creator >& )
{
    return *this;
}

template< typename Creator >
inline void WLROICtrlFactoryBase< Creator >::registerCreator( const std::string& name, Creator creator )
{
    m_creators[name] = creator;
}

template< typename Creator >
inline typename WLROICtrlFactoryBase< Creator >::citerTc WLROICtrlFactoryBase< Creator >::begin() const
{
    return m_creators.begin();
}

template< typename Creator >
inline typename WLROICtrlFactoryBase< Creator >::citerTc WLROICtrlFactoryBase< Creator >::end() const
{
    return m_creators.end();
}

template< typename Creator >
inline typename WLROICtrlFactoryBase< Creator >::citerTc WLROICtrlFactoryBase< Creator >::find( const std::string& name ) const
{
    return m_creators.find( name );
}

#endif /* WLROICTRLFACTORYBASE_H_ */
