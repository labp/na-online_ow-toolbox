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

#ifndef WLSMARTSTORAGE_H_
#define WLSMARTSTORAGE_H_

#include <ostream>
#include <vector>

#include <boost/shared_ptr.hpp>

/**
 * The WLSmartStorage represents a smart memory container.
 * All memory allocating and deallocating operations will be done by the based boost shared pointer.
 *
 * The container supports tow basic methods:
 *
 *  setData: Clears the memory container and writes only the new data into.
 *  append: Extends the memory range by the new data at the end. Old data won't deleted.
 */
class WLSmartStorage
{
public:

    /**
     * The container type.
     */
    typedef std::vector< char > ContainerT;

    /**
     * A pointer on a ContainerT.
     */
    typedef boost::shared_ptr< ContainerT > ContainerT_SPtr;

    /**
     * Constructs a new empty container.
     */
    WLSmartStorage();

    /**
     * Constructs a new container, which includes the @data.
     *
     * @param value The new data.
     */
    WLSmartStorage( std::string data );

    /**
     * Constructs a new container, which includes the @data.
     *
     * @param data The pointer to the data.
     * @param size The size of data.
     */
    WLSmartStorage( const void * data, size_t size );

    /**
     * Gets the containing data as string. This makes sense when the container contains string only data.
     *
     * @return The data string.
     */
    std::string toString() const;

    /**
     * Gets a pointer to the containers first byte.
     */
    const void *getData() const;

    /**
     * Gets a pointer to the containers first byte.
     */
    void *getData();

    /**
     * Gets the size of memory inside of the container.
     *
     * @return The size of memory.
     */
    size_t getSize() const;

    /**
     * Clears the container and deletes the content.
     */
    void clear();

    /**
     * Sets the containers content. Existing content will be deleted before.
     *
     * @param data The pointer to the data.
     * @param size The size of the data.
     */
    void setData( const void * data, size_t size );

    /**
     * Sets the containers content. Existing content will be deleted before.
     *
     * @param data The data string.
     */
    void setData( std::string data );

    /**
     * Appends new data to the container. Existing content will be untouched.
     *
     * @param data The pointer to the data.
     * @param size The size of the data.
     */
    void append( const void * data, size_t size );

    /**
     * Appends new data to the container. Existing content will be untouched.
     *
     * @param data The data string.
     */
    void append( std::string data );

private:

    /**
     * Method to clear the container by resetting the shared pointer and initializing them with a new empty container.
     */
    void init();

    /**
     * The memory container.
     */
    ContainerT_SPtr m_data;
};

/**
 * Returns an out stream containing the @store as string at its end.
 *
 * @param str The output stream.
 * @param store The WLSmartStorage object to concatenate.
 * @return The out stream.
 */
inline std::ostream& operator<<( std::ostream &str, const WLSmartStorage &store )
{
    str << store.toString();

    return str;
}

#endif /* WLSMARTSTORAGE_H_ */
