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

#ifndef WLEMDATA_H
#define WLEMDATA_H

#include <cstddef>
#include <list>
#include <ostream>
#include <stdint.h>
#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include "core/container/WLArrayList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/enum/WLECoordSystem.h"
#include "core/data/enum/WLEExponent.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/enum/WLEUnit.h"

/**
 * Class for general modality. Saves information which are present for all modalities.
 *
 * \authors kaehler, pieloth, maschke
 */
class WLEMData: public boost::enable_shared_from_this< WLEMData >
{
public:
    static const std::string CLASS;

    static const WLFreqT UNDEFINED_FREQ;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMData > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMData > ConstSPtr;

    typedef std::list< size_t > ChannelList;

    typedef boost::shared_ptr< ChannelList > ChannelListSPtr;

    /**
     * Data type of single value aka "Channel c1 at time t1".
     */
#ifdef LABP_FLOAT_COMPUTATION
    typedef float ScalarT;
#else
    typedef double ScalarT;
#endif  // LABP_FLOAT_COMPUTATION
    /**
     * Data type of a multi channel sample for a defined time point aka "All channels at time t1".
     */
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, 1 > SampleT;

    /**
     * Data type of a measured single channel over time.
     */
    typedef Eigen::Matrix< ScalarT, 1, Eigen::Dynamic > ChannelT;

    /**
     * Date type of a multi channel measurement: Channel x Time
     */
    typedef Eigen::Matrix< ScalarT, Eigen::Dynamic, Eigen::Dynamic > DataT;

    typedef boost::shared_ptr< DataT > DataSPtr;

    typedef boost::shared_ptr< const DataT > DataConstSPtr;

    /**
     * Constructor
     */
    WLEMData();

    /**
     * Constructor, sets data from input instance.
     *
     * \param emd Adopts the attributes from this instance, but not the "raw" data.
     */
    explicit WLEMData( const WLEMData& emd );

    /**
     * Destructor
     */
    virtual ~WLEMData();

    /**
     * Cast to EMD if possible. Check modality type with getModalityType()!
     *
     * \return Shared Pointer< EMD >
     */
    template< typename EMD >
    boost::shared_ptr< EMD > getAs()
    {
        return boost::dynamic_pointer_cast< EMD >( shared_from_this() );
    }

    /**
     * Cast to EMD if possible. Check modality type with getModalityType()!
     *
     * \return Shared Pointer< const EMD >
     */
    template< typename EMD >
    boost::shared_ptr< const EMD > getAs() const
    {
        return boost::dynamic_pointer_cast< EMD >( shared_from_this() );
    }

    /**
     * Clones this instance without copying the "raw" data.
     *
     * \return A copy of this instance, except the "raw" data.
     */
    virtual WLEMData::SPtr clone() const = 0;

    /**
     * Returns the data. NOTE: The method does not modify any object data, but data may modified indirectly!
     */
    virtual DataT& getData() const;

    /**
     * Returns the data without the bad channels.
     * NOTE: The method does not modify any object data, but data may modified indirectly!
     *
     * \return The data.
     */
    virtual DataSPtr getDataBadChannels() const;

    /**
     * Returns the data without the bad channels.
     * NOTE: The method does not modify any object data, but data may modified indirectly!
     *
     * \return The data.
     */
    virtual DataSPtr getDataBadChannels( ChannelListSPtr badChans ) const;

    /**
     * Sets the data.
     *
     * \param data Data to set.
     */
    virtual void setData( DataSPtr data );

    /**
     * Gets the highpass frequency.
     *
     * \return Frequency in Hz
     */
    WLFreqT getAnalogHighPass() const;

    /**
     * Sets the highpass frequency.
     *
     * \param analogHighPass Frequency in Hz
     */
    void setAnalogHighPass( WLFreqT analogHighPass );

    /**
     * Gets the lowpass frequency.
     *
     * \return Frequency in Hz
     */
    WLFreqT getAnalogLowPass() const;

    /**
     * Sets the lowpass frequency.
     *
     * \param analogLowPass Frequency in Hz
     */
    void setAnalogLowPass( WLFreqT analogLowPass );

    /**
     * Gets the channel names.
     *
     * \return Channel names.
     */
    WLArrayList< std::string >::SPtr getChanNames();

    /**
     * Gets the channel names.
     *
     * \return Channel names.
     */
    WLArrayList< std::string >::ConstSPtr getChanNames() const;

    /**
     * Sets the channel names.
     *
     * \param chanNames Names to set.
     */
    void setChanNames( WLArrayList< std::string >::SPtr chanNames );

    /**
     * Gets the unit of the data.
     *
     * \return Unit
     */
    WLEUnit::Enum getChanUnit() const;

    /**
     * Gets the exponent of the data.
     *
     * \return Exponent
     */
    WLEExponent::Enum getChanUnitExp() const;

    /**
     * Sets the exponent of the data.
     *
     * \param chanUnitExp Exponent to set
     */
    void setChanUnitExp( WLEExponent::Enum chanUnitExp );

    /**
     * Sets the unit of the data.
     *
     * \param chanUnit Unit to set
     */
    void setChanUnit( WLEUnit::Enum chanUnit );

    /**
     * Gets the coordinate system.
     *
     * \return Coordinate system
     */
    WLECoordSystem::Enum getCoordSystem() const;

    /**
     * Sets the coordinate system.
     *
     * \param coordSystem Coordinate system to set
     */
    void setCoordSystem( WLECoordSystem::Enum coordSystem );

    /**
     * Gets the line frequency.
     *
     * \return Frequency in Hz
     */
    WLFreqT getLineFreq() const;

    /**
     * Sets the line frequency.
     *
     * \param lineFreq Frequency in Hz
     */
    void setLineFreq( WLFreqT lineFreq );

    /**
     * Gets the name of the measurement device.
     *
     * \return name of measurement device
     */
    std::string getMeasurementDeviceName() const;

    /**
     * Set the name of the measurement device.
     *
     * \param measurementDeviceName Name of the measurement device
     */
    void setMeasurementDeviceName( std::string measurementDeviceName );

    /**
     * Gets the modality type of this instance.
     *
     * \return modality type
     */
    virtual WLEModality::Enum getModalityType() const = 0;

    /**
     * Gets the number of channels (dimension 1).
     *
     * \return Number of channels
     */
    virtual WLChanNrT getNrChans() const;

    /**
     * Gets the samples of a channel for this block (dimension 2).
     *
     * \return Number of Samples
     */
    virtual WLSampleNrT getSamplesPerChan() const;

    /**
     * Returns the bad channel list.
     *
     * \return Bad channel list.
     */
    virtual ChannelListSPtr getBadChannels() const;

    /**
     * Returns sampling frequency in Hz.
     *
     * \return sampling frequency in Hz
     */
    WLFreqT getSampFreq() const;

    /**
     * Sets the sampling frequency.
     *
     * \param sampFreq Frequency in Hz
     */
    void setSampFreq( WLFreqT sampFreq );

    /**
     * Returns the data length in seconds using samples and frequency.
     *
     * \return data length in seconds.
     */
    WLTimeT getLength() const;

    void setBadChannels( ChannelListSPtr badChannels );

    static std::string channelToString( const ChannelT& data, size_t maxSamples );

    static std::string dataToString( const DataT& data, size_t maxChannels, size_t maxSamples );

    /**
     * Returns true if the channel number is listed in the bad channel list.
     *
     * \param channelNo The channel number.
     * \return True / false.
     */
    bool isBadChannel( size_t channelNo ) const;

protected:
    std::string m_measurementDeviceName; /**< name of the measurement device */

    WLFreqT m_lineFreq; /**< power line frequency */

    WLArrayList< std::string >::SPtr m_chanNames;

    WLFreqT m_sampFreq; /**<sampling frequency (unique within modality) */

    WLEUnit::Enum m_chanUnit;

    WLEExponent::Enum m_chanUnitExp; /**< data is in unit m_chanUnit * 10^m_chanUnitExp */

    WLFreqT m_analogHighPass; /**< cutoff frequency of highpass filter in analog processing chain */

    WLFreqT m_analogLowPass; /**< cutoff frequency of lowpass filter in analog processing chain */

    WLECoordSystem::Enum m_CoordSystem; /**< type of coordinate system used for m_chanPositions */

    DataSPtr m_data; /**< Raw data of the measurement. */

    ChannelListSPtr m_badChannels; /**< List of the bad channels. */
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMData& obj )
{
    strm << WLEMData::CLASS << "::" << WLEModality::name( obj.getModalityType() ) << ": data=" << obj.getNrChans() << "x"
                    << obj.getSamplesPerChan();
    return strm;
}

#endif  // WLEMDATA_H
