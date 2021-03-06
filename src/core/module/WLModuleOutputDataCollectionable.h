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
 * Extends an output connector by a correct use of WModuleInputDataCollection.
 * It checks whether the input connector is an instance of WModuleInputDataCollection, if true addData() is called.
 * \attention Do not use the static methods WModuleOutputData::create() and WModuleOutputData::createAndAdd()!
 *
 * \author pieloth
 * \ingroup module
 */
template< typename T >
class WLModuleOutputDataCollectionable: public WModuleOutputData< T >
{
public:
    typedef boost::shared_ptr< WLModuleOutputDataCollectionable< T > > SPtr; //!< Abbreviation for a shared pointer.

    typedef boost::shared_ptr< const WLModuleOutputDataCollectionable< T > > ConstSPtr; //!< Abbreviation for const shared pointer.

    /**
     * Returns a new instance.
     *
     * \param module the module which is owner of this connector.
     * \param name The name of this connector.
     * \param description Short description of this connector.
     * \return new instance of WLModuleOutputDataCollectionable
     */
    static WLModuleOutputDataCollectionable< T >::SPtr instance( WModule::SPtr module, std::string name = "",
                    std::string description = "" );

    /**
     * Constructor.
     *
     * \param module the module which is owner of this connector.
     * \param name The name of this connector.
     * \param description Short description of this connector.
     */
    WLModuleOutputDataCollectionable( WModule::SPtr module, std::string name = "", std::string description = "" ) :
                    WModuleOutputData< T >( module, name, description )
    {
    }

    /**
     * Updates the data associated. For instances of WLModuleInputDataCollection addData() is called.
     *
     * \param data The data to send.
     */
    void updateData( boost::shared_ptr< T > data );

protected:
private:
};

template< typename T >
boost::shared_ptr< WLModuleOutputDataCollectionable< T > > WLModuleOutputDataCollectionable< T >::instance( WModule::SPtr module,
                std::string name, std::string description )
{
    WLModuleOutputDataCollectionable< T >::SPtr instance = WLModuleOutputDataCollectionable< T >::SPtr(
                    new WLModuleOutputDataCollectionable< T >( module, name, description ) );
    return instance;
}

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
