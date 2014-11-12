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

#ifndef WFTCHUNKREADERNEUROMAGHDR_H_
#define WFTCHUNKREADERNEUROMAGHDR_H_

#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <fiff/fiff_info.h>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/emd/WLEMDRaw.h"
#include "core/data/enum/WLEModality.h"
#include "WFTChunkReader.h"

/**
 * Reads the Neuromag header "FT_CHUNK_NEUROMAG_HEADER", which contains a fif file.
 * It creates the measurement information object for online data processing.
 *
 * \authors maschke, pieloth
 */
class WFTChunkReaderNeuromagHdr: public WFTChunkReader
{
public:
    /**
     * A shared pointer on a WFTChunkReaderNeuromagHdr.
     */
    typedef boost::shared_ptr< WFTChunkReaderNeuromagHdr > SPtr;

    /**
     * A shared pointer on a constant WFTChunkReaderNeuromagHdr.
     */
    typedef boost::shared_ptr< const WFTChunkReaderNeuromagHdr > ConstSPtr;

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

    WFTChunkReaderNeuromagHdr();

    virtual ~WFTChunkReaderNeuromagHdr();

    virtual wftb::chunk_type_t supportedChunkType() const;

    virtual bool read( const WFTChunk::ConstSPtr chunk );

    virtual bool apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw );

    /**
     * Gets the measurement information.
     *
     * \return Returns a pointer to a constant measurement information.
     */
    MeasInfo_ConstSPtr getMeasInfo() const;

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

    /**
     * Extracts the event/ stimulus channels from a data matrix. @ePicks contains the needed channel indices.
     *
     * \param rawData The data matrix.
     * \param ePicks A vector contains the event channel indices.
     *
     * \return Returns a pointer on the event channel matrix.
     */
    boost::shared_ptr< WLEMMeasurement::EDataT > readEventChannels( const Eigen::MatrixXf& rawData,
                    WLEMDRaw::ChanPicksT ePicks ) const;

private:
    /**
     * The measurement information.
     */
    MeasInfo_SPtr m_measInfo;

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

#endif  // WFTCHUNKREADERNEUROMAGHDR_H_
