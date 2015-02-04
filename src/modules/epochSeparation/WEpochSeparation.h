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

#ifndef WEPOCHSEPARATION_H_
#define WEPOCHSEPARATION_H_

#include <cstddef>
#include <deque>
#include <list>
#include <string>
#include <set>

#include <core/common/WException.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/util/WLRingBuffer.h"

/**
 * Epoch separation based on trigger detection. Epochs can be combined from several packets.
 * A combined epoch contains the checked event channel only!
 */
class WEpochSeparation
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochSeparation > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WEpochSeparation > ConstSPtr;

    static const std::string CLASS;

    WEpochSeparation();

    WEpochSeparation( WLChanIdxT channel, std::set< WLEMMeasurement::EventT > triggerMask, WLSampleNrT preSamples,
                    WLSampleNrT postSamples );

    virtual ~WEpochSeparation();

    // getter & setter //

    /**
     * Returns the chosen index of the event channel.
     */
    WLChanIdxT getChannel() const;

    /**
     * Sets the index to be used for the event channel.
     */
    void setChannel( WLChanIdxT channel );

    /**
     * Returns the trigger mask to be tested.
     */
    std::set< WLEMMeasurement::EventT > getTriggerMask() const;

    /**
     * Sets the trigger mask, which will be checked on the event channel.
     */
    void setTriggerMask( std::set< WLEMMeasurement::EventT > triggerMask );

    /**
     * Returns the samples to be stored before the trigger.
     */
    WLSampleNrT getPreSamples() const;

    /**
     * Sets the samples count, which shall be stored before the trigger.
     */
    void setPreSamples( WLSampleNrT preSamples );

    /**
     * Returns the samples to be stored after the trigger.
     */
    WLSampleNrT getPostSamples() const;

    /**
     * Sets the samples count, which shall be stored after the trigger.
     */
    void setPostSamples( WLSampleNrT postSamples );

    /**
     * Reset all necessary attributes.
     */
    void reset();

    /**
     * True if completely separated epochs are available.
     *
     * \return True if completely separated epochs are available.
     */
    bool hasEpochs() const;

    /**
     * Returns the count of completely separated epochs.
     *
     * \return Count of completely separated epochs.
     */
    size_t epochSize() const;

    // main methods //

    /**
     * Checks the passed in EMM for an event with the trigger mask on the set up channel.
     *
     * \return Number of extracted epochs
     */
    size_t extract( const WLEMMeasurement::SPtr emmIn ); // TODO(pieloth): ConstSPtr conflicts with WLRingBuffer

    /**
     * Returns the  next extracted EMM.This is a new object which contains the checked event channel only (EventChannelSize == 1)!
     * The EMM is returned only once - FIFO!
     *
     * \return EMM object.
     */
    WLEMMeasurement::SPtr getNextEpoch();

private:
    /**
     * Data structure to store not completely collected epochs.
     */
    class LeftEpoch
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< LeftEpoch > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const LeftEpoch > ConstSPtr;

        /**
         * EMM which is currently separated
         */
        WLEMMeasurement::SPtr m_emm;

        /**
         * Left samples which has to be collected from next packets.
         */
        WLSampleNrT m_leftSamples;

        /**
         * Start index in next packet.
         */
        WLSampleIdxT m_startIndex;
    };

    /**
     * Initializes the buffer once to store past packets.
     *
     * \param emd Modality is used to retrieve block size.
     */
    void setupBuffer( WLEMData::ConstSPtr emd );

    /**
     * Creates a new EMM object and copies all past data starting from the given index (past to index).
     *
     * \param eIndex Index to start a backward copy.
     *
     * \return Structure which contains a new EMM object and left sample count.
     */
    LeftEpoch::SPtr processPreSamples( WLSampleIdxT eIndex ) throw( WException );

    /**
     * Fills the incomplete epoch from given EMM object.
     *
     * \param leftEpoch Epoch to be filled
     * \param emmIn EMM with data
     *
     * \return true, if no samples left and EMM is complete.
     */
    bool processPostSamples( LeftEpoch::SPtr leftEpoch, WLEMMeasurement::ConstSPtr emmIn );

    WLChanIdxT m_channel;
    std::set< WLEMMeasurement::EventT > m_triggerMask;
    WLSampleNrT m_preSamples; //!< Samples before trigger/event.
    WLSampleNrT m_postSamples; //!< Samples after triffer/event.
    WLSampleNrT m_blockSize; //!< Block size in samples.

    WLRingBuffer< WLEMMeasurement >::SPtr m_buffer;
    std::deque< WLEMMeasurement::SPtr > m_epochs;
    std::list< LeftEpoch::SPtr > m_leftEpochs;

    /**
     * Correct modulo calculation for negative values.
     */
    WLSampleIdxT nmod( WLSampleIdxT a, WLSampleNrT n );
};

#endif  // WEPOCHSEPARATION_H_
