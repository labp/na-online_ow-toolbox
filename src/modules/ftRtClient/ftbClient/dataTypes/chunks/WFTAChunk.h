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

#ifndef WFTACHUNK_H_
#define WFTACHUNK_H_

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/ftbClient/container/WLSmartStorage.h"
#include "modules/ftRtClient/ftbClient/dataTypes/enum/WLEFTChunkType.h"

/**
 * The abstract WFTAChunk class defines a generic interface class for all FieldTrip header chunk classes.
 * Any deriving class has to convert the data pointer into the specific data structure. This will be done
 * by the method @process called by the constructor.
 */
class WFTAChunk: public boost::enable_shared_from_this< WFTAChunk >
{
public:
    /**
     * A shared pointer on a WFTAChunk.
     */
    typedef boost::shared_ptr< WFTAChunk > SPtr;

    /**
     * A shared pointer on a constant WFTAChunk.
     */
    typedef boost::shared_ptr< const WFTAChunk > ConstSPtr;

    /**
     * Constructs a new WFTAChunk and processes the memory into the generic chunk data structure.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     */
    explicit WFTAChunk( WLEFTChunkType::Enum type, const size_t size );

    /**
     * Destroys the WFTAChunk.
     */
    virtual ~WFTAChunk();

    /**
     * Determines whether the chunk was processed successfully.
     *
     * \return Returns true if the process was successful, otherwise false.
     */
    bool isValid() const;

    /**
     * Gets the chunks buffer size.
     *
     * \return Returns the chunks buffer size.
     */
    size_t getSize() const;

    /**
     * Gets the chunk type.
     *
     * \return Returns the chunk type.
     */
    WLEFTChunkType::Enum getType() const;

    /**
     * Gets the data as a smart storage structure. This method is used to serialize a chunk into a request message body.
     *
     * \return Returns a shared pointer on a constant smart storage.
     */
    virtual WLSmartStorage::ConstSPtr serialize() const = 0;

    /**
     * Gets the chunks as desired pointer.
     *
     * \return Returns the chunk as shared pointer.
     */
    template< typename Chunk >
    boost::shared_ptr< Chunk > getAs()
    {
        return boost::dynamic_pointer_cast< Chunk >( shared_from_this() );
    }

    /**
     * Gets the chunks as desired pointer.
     *
     * \return Returns the chunk as shared pointer.
     */
    template< typename Chunk >
    boost::shared_ptr< const Chunk > getAs() const
    {
        return boost::dynamic_pointer_cast< Chunk >( shared_from_this() );
    }

protected:
    /**
     * Based on the stored memory of @data, this method creates the chunks data structure.
     * It has to implement by a deriving class for a special chunk type.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     * \return Returns true if the processing was successful, otherwise false.
     */
    virtual bool process( const char* data, size_t size ) = 0;

    /**
     * Private method to call the pure virtual method "process()" not directly at the constructor.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     */
    void processData( const char* data, const size_t size );

private:
    /**
     * Determines the validation of the chunk.
     */
    bool m_valid;

    /**
     * The chunks buffer size.
     */
    size_t m_size;

    /**
     * The chunk type
     */
    WLEFTChunkType::Enum m_type;
};

#endif  // WFTACHUNK_H_
