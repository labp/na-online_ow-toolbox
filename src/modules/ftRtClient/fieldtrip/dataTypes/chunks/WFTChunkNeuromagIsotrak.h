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

#ifndef WFTCHUNKNEUROMAGISOTRAK_H_
#define WFTCHUNKNEUROMAGISOTRAK_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/enum/WLEPointType.h"
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
     * A vector describing faces for the 3D view.
     */
    typedef WLArrayList< WVector3i > FacesT;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * A calculation factor when creating the 3D EEG faces.
     */
    static const int EEG_FACES_FACTOR;

    /**
     * Constructs a new WFTChunkNeuromagIsotrak.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     */
    explicit WFTChunkNeuromagIsotrak( const char* data, const size_t size );

    /**
     * Gets the data as a smart storage structure. This method is used to serialize a chunk into a request message body.
     *
     * Inherited method from WFTAChunk.
     *
     * \return Returns a shared pointer on a constant smart storage.
     */
    WLSmartStorage::ConstSPtr serialize() const;

    /**
     * Gets the digitalization points list.
     *
     * \return Returns a pointer to a constant digitalization points list.
     */
    WLList< WLDigPoint >::SPtr getDigPoints() const;

    /**
     * Gets the digitalization points list for the @type.
     *
     * \param type The type of dig. points.
     * \return Returns a shared pointer on a digitalization points list.
     */
    WLList< WLDigPoint >::SPtr getDigPoints( WLEPointType::Enum type ) const;

    /**
     * Gets the EEG channel positions.
     *
     * \return Returns a shared pointer on a WLArrayList< WPosition >.
     */
    WLArrayList< WPosition >::SPtr getEEGChanPos() const;

    /**
     * Gets the EEG 3D faces.
     *
     * \return Returns a shared pointer on a WLArrayList< WVector3i >.
     */
    WLArrayList< WVector3i >::SPtr getEEGFaces() const;

protected:
    /**
     * Based on the stored memory of @data, this method creates the chunks data structure.
     * It has to implement by a deriving class for a special chunk type.
     *
     * Inherited method from WFTAChunk.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     *
     * \return Returns true if the processing was successful, otherwise false.
     */
    bool process( const char* data, size_t size );

    /**
     * Creates the EEG channel positions from the isotak digitalization points.
     *
     * \param digPoints The list of digitalization points.
     * \return Returns true if the channel positions were created, otherwise false.
     */
    bool createEEGPositions( WLList< WLDigPoint >::ConstSPtr digPoints );

    /**
     * The digitalization points list.
     */
    WLList< WLDigPoint >::SPtr m_digPoints;

    /**
     * The EEG channel positions.
     */
    WLArrayList< WPosition >::SPtr m_eegChPos;

    /**
     * The EEG faces.
     */
    WLArrayList< WVector3i >::SPtr m_eegFaces;
};

#endif  // WFTCHUNKNEUROMAGISOTRAK_H_
