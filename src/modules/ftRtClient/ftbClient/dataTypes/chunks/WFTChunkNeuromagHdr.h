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

#ifndef WFTCHUNKNEUROMAGHDR_H_
#define WFTCHUNKNEUROMAGHDR_H_

#include <map>
#include <string>
#include <vector>

#include <fiff/fiff_info.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/emd/WLEMDRaw.h"
#include "core/data/enum/WLEModality.h"
#include "WFTAChunk.h"

/**
 * The WFTChunkNeuromagHdr represents the FieldTrip header chunk, which contains the Neuromag header file.
 * It creates the measurement information object for online data processing.
 *
 * \author maschke
 */
class WFTChunkNeuromagHdr: public WFTAChunk
{
public:
    /**
     * A shared pointer on a WFTChunkNeuromagHdr.
     */
    typedef boost::shared_ptr< WFTChunkNeuromagHdr > SPtr;

    /**
     * A shared pointer on a constant WFTChunkNeuromagHdr.
     */
    typedef boost::shared_ptr< const WFTChunkNeuromagHdr > ConstSPtr;

    /**
     * A shared pointer on a measurement information.
     */
    typedef boost::shared_ptr< FIFFLIB::FiffInfo > MeasInfo_SPtr;

    /**
     * A shared pointer on a constant measurement information.
     */
    typedef boost::shared_ptr< const FIFFLIB::FiffInfo > MeasInfo_ConstSPtr;

    /**
     * A map with a modality type as key and a row vector as mapped type.
     */
    typedef std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT > ModalityPicksT;

    /**
     * A shared pointer on a ModalityPicksT.
     */
    typedef boost::shared_ptr< ModalityPicksT > ModalityPicks_SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WFTChunkNeuromagHdr and processes the memory into the measurement information structure.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     */
    explicit WFTChunkNeuromagHdr( const char* data, const wftb::chunk_size_t size );

    /**
     * Gets the data as a smart storage structure. This method is used to serialize a chunk into a request message body.
     *
     * Inherited method from WFTAChunk.
     *
     * \return Returns a shared pointer on a constant smart storage.
     */
    WLSmartStorage::ConstSPtr serialize() const;

    /**
     * Gets the measurement information.
     *
     * \return Returns a pointer to a constant measurement information.
     */
    MeasInfo_ConstSPtr getData() const;

    /**
     * Gets the channel names for the @modality if they exist.
     *
     * \param modality The modality type.
     * \return Returns a shared pointer on a constant string list.
     */
    WLArrayList< std::string >::SPtr getChannelNames( WLEModality::Enum modality ) const;

    /**
     * Gets the modality picks.
     *
     * \return Returns a shared pointer on a std::map.
     */
    ModalityPicks_SPtr getModalityPicks() const;

    /**
     * Gets the stimulus picks vector.
     *
     * \return Returns a shared pointer on a row vector.
     */
    boost::shared_ptr< WLEMDRaw::ChanPicksT > getStimulusPicks() const;

    /**
     * Gets the channel positions for EEG.
     *
     * \return Returns a shared pointer on a channel position vector.
     */
    WLArrayList< WPosition >::SPtr getChannelPositionsEEG() const;

    /**
     * Gets the channel points for MEG.
     *
     * \return Returns a shared pointer on a channel position vector.
     */
    WLArrayList< WPosition >::SPtr getChannelPositionsMEG() const;

    /**
     * Gets the x-axis unit vector for coil coordinate system.
     *
     * \return Returns a shared pointer on a vector.
     */
    WLArrayList< WVector3f >::SPtr getChannelExMEG() const;

    /**
     * Gets the y-axis unit vector for coil coordinate system.
     *
     * \return Returns a shared pointer on a vector.
     */
    WLArrayList< WVector3f >::SPtr getChannelEyMEG() const;

    /**
     * Gets the z-axis unit vector for coil coordinate system.
     *
     * \return Returns a shared pointer on a vector.
     */
    WLArrayList< WVector3f >::SPtr getChannelEzMEG() const;

    /**
     * Gets the scaling factors.
     *
     * \return Returns a shared pointer on a float vector.
     */
    boost::shared_ptr< std::vector< float > > getScaleFactors() const;

    /**
     * Gets whether or not there exists channel position information for the EEG system.
     *
     * \return Returns true if there are channel positions, otherwise false.
     */
    bool hasChannelPositionsEEG() const;

    /**
     * Gets whether or not there exists channel position information for the MEG system.
     *
     * \return Returns true if there are channel positions, otherwise false.
     */
    bool hasChannelPositionsMEG() const;

protected:
    /**
     * Based on the stored memory of @data, this method creates the chunks data structure.
     * It has to implement by a deriving class for a special chunk type.
     *
     * Inherited method from WFTAChunk.
     *
     * \param data The memory storage, which contains the chunk data.
     * \param size The size of the memory storage.
     * \return Returns true if the processing was successful, otherwise false.
     */
    bool process( const char* data, size_t size );

    /**
     * The measurement information.
     */
    MeasInfo_SPtr m_measInfo;

private:
    /**
     * A row vector, which contains the channel indices of the event/ stimulus channels.
     */
    boost::shared_ptr< WLEMDRaw::ChanPicksT > m_stimulusPicks;

    ModalityPicks_SPtr m_modalityPicks; /**< A map, which contains the channel indices for each modality type. */

    WLArrayList< WPosition >::SPtr m_chPosEEG; /**< The channel positions for EEG. */

    WLArrayList< WPosition >::SPtr m_chPosMEG; /**< The channel position for MEG. */

    WLArrayList< WVector3f >::SPtr m_chExMEG; /**< Coil coordinate system x-axis unit vector. */
    WLArrayList< WVector3f >::SPtr m_chEyMEG; /**< Coil coordinate system y-axis unit vector. */
    WLArrayList< WVector3f >::SPtr m_chEzMEG; /**< Coil coordinate system z-axis unit vector. */

    boost::shared_ptr< std::vector< float > > m_scaleFactors; /**< Vector for scaling factors. */
};

#endif  // WFTCHUNKNEUROMAGHDR_H_
