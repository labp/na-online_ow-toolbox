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

#ifndef WEPOCHSEPARATION_H_
#define WEPOCHSEPARATION_H_

#include <cstddef>
#include <deque>
#include <list>
#include <string>
#include <set>

#include "core/data/WLEMMeasurement.h"
#include "core/util/WLRingBuffer.h"

/**
 * Epoch separation based on trigger detection. Epochs can be combined from several packets.
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

    WEpochSeparation( size_t channel, std::set< WLEMMeasurement::EventT > triggerMask, size_t preSamples, size_t postSamples );

    virtual ~WEpochSeparation();

    // getter & setter //

    /**
     * Returns the chosen index of the event channel.
     */
    size_t getChannel() const;

    /**
     * Sets the index to be used for the event channel.
     */
    void setChannel( size_t channel );

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
    size_t getPreSamples() const;

    /**
     * Sets the samples count, which shall be stored before the trigger.
     */
    void setPreSamples( size_t preSamples );

    /**
     * Returns the samples to be stored after the trigger.
     */
    size_t getPostSamples() const;

    /**
     * Sets the samples count, which shall be stored after the trigger.
     */
    void setPostSamples( size_t postSamples );

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
     * Checks the passed in WDataSetEMM for an event with the trigger mask on the set up channel.
     * Returns Number of extracted epochs
     */
    size_t extract( const WLEMMeasurement::SPtr emmIn ); // TODO(pieloth): ConstSPtr conflicts with WLRingBuffer

    /**
     * Returns the extracted WDataSetEMM. This is a new object. (FIFO)
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
        size_t m_leftSamples;

        /**
         * Start index in next packet.
         */
        size_t m_startIndex;
    };

    /**
     * Initializes the buffer once to store past packets.
     *
     * \param emd Modality is used to retrieve block size.
     */
    void setupBuffer( LaBP::WLEMD::ConstSPtr emd );

    /**
     * Creates a new EMM object and copies all past data starting from the given index (past to index).
     *
     * \param eIndex Index to start a backward copy.
     *
     * \return Structure which contains a new EMM object and left sample count.
     */
    LeftEpoch::SPtr processPreSamples( size_t eIndex );

    /**
     * Fills the incomplete epoch from given EMM object.
     *
     * \param leftEpoch Epoch to be filled
     * \param emmIn EMM with data
     *
     * \return true, if no samples left and EMM is complete.
     */
    bool processPostSamples( LeftEpoch::SPtr leftEpoch, WLEMMeasurement::ConstSPtr emmIn );

    size_t m_channel;
    std::set< WLEMMeasurement::EventT > m_triggerMask;
    size_t m_preSamples;
    size_t m_postSamples;
    size_t m_blockSize;

    LaBP::WLRingBuffer< WLEMMeasurement >::SPtr m_buffer;
    std::deque< WLEMMeasurement::SPtr > m_epochs;
    std::list< LeftEpoch::SPtr > m_leftEpochs;

    /**
     * Correct modulo calculation for negative values.
     */
    size_t nmod( ptrdiff_t a, size_t n );
};

#endif  // WEPOCHSEPARATION_H_
