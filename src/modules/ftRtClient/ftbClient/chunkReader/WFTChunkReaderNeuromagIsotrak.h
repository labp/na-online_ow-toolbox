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

#ifndef WFTCHUNKREADERNEUROMAGISOTRAK_H_
#define WFTCHUNKREADERNEUROMAGISOTRAK_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WMatrixFixed.h>
#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/WLPositions.h"
#include "core/data/enum/WLEPointType.h"
#include "core/data/WLDigPoint.h"
#include "WFTChunkReader.h"

/**
 * Reads the Neuromag header "FT_CHUNK_NEUROMAG_ISOTRAK", which contains a fif file.
 * It creates the digitalization points for online data processing.
 *
 * \authors maschke, pieloth
 */
class WFTChunkReaderNeuromagIsotrak: public WFTChunkReader
{
public:
    /**
     * A shared pointer on a WFTChunkReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< WFTChunkReaderNeuromagIsotrak > SPtr;

    /**
     * A shared pointer on a constant WFTChunkReaderNeuromagIsotrak.
     */
    typedef boost::shared_ptr< const WFTChunkReaderNeuromagIsotrak > ConstSPtr;

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

    WFTChunkReaderNeuromagIsotrak();

    virtual ~WFTChunkReaderNeuromagIsotrak();

    virtual bool read( WFTChunk::ConstSPtr chunk );

    virtual bool apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw );

    virtual wftb::chunk_type_t supportedChunkType() const;

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
     * \return Returns a shared pointer on a WLPositions.
     */
    WLPositions::SPtr getEEGChanPos() const;

    /**
     * Gets the EEG 3D faces.
     *
     * \return Returns a shared pointer on a WLArrayList< WVector3i >.
     */
    WLArrayList< WVector3i >::SPtr getEEGFaces() const;

private:
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
    WLPositions::SPtr m_eegChPos;

    /**
     * The EEG faces.
     */
    WLArrayList< WVector3i >::SPtr m_eegFaces;
};

#endif  // WFTCHUNKREADERNEUROMAGISOTRAK_H_
