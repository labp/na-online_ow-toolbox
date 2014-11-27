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

#ifndef WLMODULEINPUTDATACOLLECTION_H
#define WLMODULEINPUTDATACOLLECTION_H

#include <cstddef>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WException.h>
#include <core/kernel/WModule.h>
#include <core/kernel/WModuleInputConnector.h>

/**
 * Class offering an instantiateable data connection between modules.
 * This abstract class can implement various collections e.g. to provide an input buffer for a module.
 * All implementations must be used together with WLModuleOutputDataCollectionable.
 *
 * \author pieloth
 * \ingroup module
 */
template< typename T >
class WLModuleInputDataCollection: public WModuleInputConnector
{
public:
    /**
     * Shared pointer to this class.
     */
    typedef boost::shared_ptr< WLModuleInputDataCollection > SPtr;

    /**
     * Const shared pointer to this class.
     */
    typedef boost::shared_ptr< const WLModuleInputDataCollection > ConstSPtr;

    /**
     * Constructor.
     *
     * \param module the module which is owner of this connector.
     * \param name The name of this connector.
     * \param description Short description of this connector.
     */
    WLModuleInputDataCollection( WModule::SPtr module, std::string name, std::string description ) :
                    WModuleInputConnector( module, name, description )
    {
    }

    /**
     * Destructor.
     */
    virtual ~WLModuleInputDataCollection()
    {
    }

    /**
     * Gives one data element and resets the update flag. Which element is returned depends on the implementation.
     *
     * \param reset resets the flag of updated() if true (default).
     * \return a data element or throws an exception if no data is available.
     */
    virtual const boost::shared_ptr< T > getData( bool reset = true ) throw( WException ) = 0;

    /**
     * Adds an element to this collection.
     *
     * \param value element whose presence in this collection is to be ensured.
     * \return true if this collection holds the element.
     */
    virtual bool addData( boost::shared_ptr< T > value ) = 0;

    /**
     * Removes all elements from this collection.
     */
    virtual void clear() = 0;

    /**
     * Checks whether the collection is empty or not.
     *
     * \return true if this collection contains no elements.
     */
    virtual bool isEmpty() = 0;

    /**
     * Returns the current number of elements in this collection.
     *
     * \return current number of elements in this collection.
     */
    virtual size_t size() = 0;
};

#endif  // WLMODULEINPUTDATACOLLECTION_H
