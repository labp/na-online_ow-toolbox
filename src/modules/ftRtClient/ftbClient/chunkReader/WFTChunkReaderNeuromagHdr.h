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

    static const std::string CLASS;

    WFTChunkReaderNeuromagHdr();

    virtual ~WFTChunkReaderNeuromagHdr();

    virtual wftb::chunk_type_t supportedChunkType() const;

    virtual bool read( const WFTChunk::ConstSPtr chunk );

    virtual bool apply( WLEMMeasurement::SPtr emm, WLEMDRaw::SPtr raw );

    void setApplyScaling( bool apply );

private:
    /**
     * A map with a modality type as key and a row vector as mapped type.
     */
    typedef std::map< WLEModality::Enum, WLEMDRaw::ChanPicksT > ModalityPicksT;

    /**
     * A map with a modality type as key and a string list as mapped type.
     */
    typedef std::map< WLEModality::Enum, WLArrayList< std::string >::SPtr > ModalityChNamesT;

    FIFFLIB::FiffInfo::SPtr m_measInfo; /**< The fiff measurement information.*/

    /**
     * A row vector, which contains the channel indices of the event/ stimulus channels.
     */
    WLEMDRaw::ChanPicksT m_stimulusPicks; /**< Contains channel indices for event/stimuli channels. */
    ModalityPicksT m_modalityPicks; /**< Contains channel indices for each modality. */
    ModalityChNamesT m_modalityChNames; /**< Contains channel names for each modality. */

    WLArrayList< WPosition >::SPtr m_chPosEEG; /**< The channel positions for EEG. */
    WLArrayList< WPosition >::SPtr m_chPosMEG; /**< The channel position for MEG. */

    WLArrayList< WVector3f >::SPtr m_chExMEG; /**< Coil coordinate system x-axis unit vector. */
    WLArrayList< WVector3f >::SPtr m_chEyMEG; /**< Coil coordinate system y-axis unit vector. */
    WLArrayList< WVector3f >::SPtr m_chEzMEG; /**< Coil coordinate system z-axis unit vector. */

    std::vector< float > m_scaleFactors; /**< Vector for scaling factors. */

    bool m_applyScaling;

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

    /**
     * Extracts and creates modality data from raw data.
     *
     * \param emm EMM to add the modalities.
     * \param raw Raw data.
     *
     * \return True if any modality was extracted.
     */
    bool extractEmdsByPicks( WLEMMeasurement::SPtr emm, WLEMDRaw::ConstSPtr raw );
};

#endif  // WFTCHUNKREADERNEUROMAGHDR_H_
