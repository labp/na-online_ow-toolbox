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

#ifndef WLMODULEOUTPUTDATACOLLECTIONABLE_H
#define WLMODULEOUTPUTDATACOLLECTIONABLE_H

#include <set>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/kernel/WModule.h>
#include <core/kernel/WModuleConnector.h>
#include <core/kernel/WModuleOutputData.h>

#include "WLModuleInputDataCollection.h"

/**
 * Class offering an instantiate-able data connection between modules.
 * This class checks whether the input connector is an instance of WModuleInputDataCollection, if true addData() is called.
 * ATTENTION: Do not use the static methods create() and createAndAdd()!
 */
template< typename T >
class WLModuleOutputDataCollectionable: public WModuleOutputData< T >
{
public:
    /**
     * Pointer to this. For convenience.
     */
    typedef boost::shared_ptr< WLModuleOutputDataCollectionable< T > > SPtr;

    /**
     * Pointer to this. For convenience.
     */
    typedef boost::shared_ptr< const WLModuleOutputDataCollectionable< T > > ConstSPtr;

    /**
     * Constructor.
     *
     * \param module the module which is owner of this connector.
     * \param name The name of this connector.
     * \param description Short description of this connector.
     */
    WLModuleOutputDataCollectionable( boost::shared_ptr< WModule > module, std::string name = "", std::string description = "" ) :
                    WModuleOutputData< T >( module, name, description )
    {
    }

    /**
     * Updates the data associated. addData() will be called for instances of WLModuleInputDataCollection.
     *
     * \param data the data do send
     */
    void updateData( boost::shared_ptr< T > data );

protected:
private:
};

template< typename T >
void WLModuleOutputDataCollectionable< T >::updateData( boost::shared_ptr< T > data )
{
    WModuleOutputData< T >::m_data = data;
    boost::shared_ptr< WLModuleInputDataCollection< T > > in;

    boost::shared_lock< boost::shared_mutex > rlock( WModuleOutputData< T >::m_connectionListLock );
    for( std::set< boost::shared_ptr< WModuleConnector > >::iterator it = WModuleOutputData< T >::m_connected.begin();
                    it != WModuleOutputData< T >::m_connected.end(); ++it )
    {
        if( ( *it )->isInputConnector() )
        {
            in = boost::dynamic_pointer_cast< WLModuleInputDataCollection< T > >( ( *it ) );
            if( in )
            {
                if( !( in->addData( WModuleOutputData< T >::m_data ) ) )
                {
                    wlog::warn( "WLModuleOutputDataCollectionable" ) << "updateData(): data skipped!";
                }
            }
        }
    }
    rlock.unlock();

    WModuleOutputData< T >::triggerUpdate();
}

#endif  // WLMODULEOUTPUTDATACOLLECTIONABLE_H
