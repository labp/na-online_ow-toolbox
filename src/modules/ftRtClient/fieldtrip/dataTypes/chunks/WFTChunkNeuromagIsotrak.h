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

#ifndef WFTCHUNKNEUROMAGISOTRAK_H_
#define WFTCHUNKNEUROMAGISOTRAK_H_

#include <boost/shared_ptr.hpp>

#include "core/container/WLList.h"
#include "core/data/WLDigPoint.h"
#include "WFTAChunk.h"

/**
 * The WFTChunkNeuromagIsotrak represents the FieldTrip header chunk, which contains the Neuromag Isotrak file.
 * It creates the digitalization points for online data processing.
 */
class WFTChunkNeuromagIsotrak: public WFTAChunk
{
public:

    /**
     * A shared pointer on a WFTChunkNeuromagIsotrak.
     */
    typedef boost::shared_ptr< WFTChunkNeuromagIsotrak > SPtr;

    /**
     * A shared pointer on a constant WFTChunkNeuromagIsotrak.
     */
    typedef boost::shared_ptr< const WFTChunkNeuromagIsotrak > Const_SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WFTChunkNeuromagIsotrak.
     *
     * @param data The memory storage, which contains the chunk data.
     * @param size The size of the memory storage.
     */
    explicit WFTChunkNeuromagIsotrak( const char* data, const size_t size );

    /**
     * Gets the chunk type.
     *
     * @return Returns the chunk type.
     */
    WLEFTChunkType::Enum getType() const;

    /**
     * Gets the digitalization points list.
     *
     * @return Returns a pointer to a constant digitalization points list.
     */
    WLList< WLDigPoint >::SPtr getData() const;

protected:

    /**
     * The path to a temporary directory. It is platform dependent.
     */
    static const std::string TMPDIRPATH;

    /**
     * The name of the temporary Neuromag Isotrak FIFF file.
     */
    static const std::string TMPFILENAME;

    /**
     * Based on the stored memory of @data, this method creates the chunks data structure.
     * It has to implement by a deriving class for a special chunk type.
     *
     * Inherited method from WFTAChunk.
     *
     * @param data The memory storage, which contains the chunk data.
     * @param size The size of the memory storage.
     * @return Returns true if the processing was successful, otherwise false.
     */
    bool process( const char* data, size_t size );

    /**
     * The digitalization points list.
     */
    WLList< WLDigPoint >::SPtr m_digPoints;
};

#endif /* WFTCHUNKNEUROMAGISOTRAK_H_ */
