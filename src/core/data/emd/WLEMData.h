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

#ifndef WLEMDATA_H
#define WLEMDATA_H

#include <cstddef>
#include <ostream>
#include <stdint.h>
#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Core>

#include <core/common/WDefines.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLEMMEnumTypes.h"

/**
 * Class for general modality. Saves information which are present for all modalities.
 */
class WLEMData: public boost::enable_shared_from_this< WLEMData >
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMData > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMData > ConstSPtr;

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
     * TODO(kaehler): Comments
     */
    WLEMData();

    explicit WLEMData( const WLEMData& emd );

    /**
     * TODO(kaehler): Comments
     */
    virtual ~WLEMData();

    /**
     * Cast to EMD if possible. Check modality type with getModalityType()!
     *
     * @return Shared Pointer< EMD >
     */
    template< typename EMD >
    boost::shared_ptr< EMD > getAs()
    {
        return boost::dynamic_pointer_cast< EMD >( shared_from_this() );
    }

    /**
     * Cast to EMD if possible. Check modality type with getModalityType()!
     *
     * @return Shared Pointer< const EMD >
     */
    template< typename EMD >
    boost::shared_ptr< const EMD > getAs() const
    {
        return boost::dynamic_pointer_cast< EMD >( shared_from_this() );
    }

    /**
     * TODO(kaehler): Comments
     */
    virtual WLEMData::SPtr clone() const = 0;

    /**
     * Returns the data. NOTE: The method does not modify any object data, but data may modified indirectly!
     */
    virtual DataT& getData() const;

    /**
     * TODO(kaehler): Comments
     */
    virtual void setData( DataSPtr data );

    /**
     * TODO(kaehler): Comments
     */
    float getAnalogHighPass() const;

    /**
     * TODO(kaehler): Comments
     */
    float getAnalogLowPass() const;

    WLArrayList< std::string >::SPtr getChanNames();

    WLArrayList< std::string >::ConstSPtr getChanNames() const;

    /**
     * TODO(kaehler): Comments
     */
    LaBP::WEUnit::Enum getChanUnit() const;

    /**
     * TODO(kaehler): Comments
     */
    LaBP::WEExponent::Enum getChanUnitExp() const;

    /**
     * TODO(kaehler): Comments
     */
    LaBP::WECoordSystemName::Enum getCoordSystem() const;

    /**
     * TODO(kaehler): Comments
     */
    uint32_t getDataBuffSizePerChan() const;

    /**
     * TODO(kaehler): Comments
     */
    uint32_t getDataOffsetIdx() const;

    /**
     * TODO(kaehler): Comments
     */
    float getLineFreq();

    /**
     * TODO(kaehler): Comments
     */
    std::string getMeasurementDeviceName() const;

    /**
     * TODO(kaehler): Comments
     */
    virtual LaBP::WEModalityType::Enum getModalityType() const = 0;

    /**
     * TODO(kaehler): Comments
     */
    virtual size_t getNrChans() const;

    virtual size_t getSamplesPerChan() const;

    /**
     * TODO(kaehler): Comments
     */
    uint16_t *getOrigIdx() const;

    /**
     * Returns sampling frequency in Hz.
     *
     * @return sampling frequency in Hz
     */
    float getSampFreq() const;

    /**
     * Returns the data length in seconds using samples and frequency.
     *
     * @return data length in seconds.
     */
    float getLength() const;

    /**
     * TODO(kaehler): Comments
     */
    void setAnalogHighPass( float analogHighPass );

    /**
     * TODO(kaehler): Comments
     */
    void setAnalogLowPass( float analogLowPass );

    void setChanNames( WLArrayList< std::string >::SPtr chanNames );

    OW_API_DEPRECATED
    void setChanNames( boost::shared_ptr< std::vector< std::string > > chanNames );

    /**
     * TODO(kaehler): Comments
     */
    void setChanUnit( LaBP::WEUnit::Enum chanUnit );

    /**
     * TODO(kaehler): Comments
     */
    void setChanUnitExp( LaBP::WEExponent::Enum chanUnitExp );

    /**
     * TODO(kaehler): Comments
     */
    void setCoordSystem( LaBP::WECoordSystemName::Enum coordSystem );

    /**
     * TODO(kaehler): Comments
     */
    void setDataBuffSizePerChan( uint32_t dataBuffSizePerChan );

    /**
     * TODO(kaehler): Comments
     */
    void setDataOffsetIdx( uint32_t dataOffsetIdx );

    /**
     * TODO(kaehler): Comments
     */
    void setLineFreq( float lineFreq );

    /**
     * TODO(kaehler): Comments
     */
    void setMeasurementDeviceName( std::string measurementDeviceName );

    /**
     * TODO(kaehler): Comments
     */
    void setOrigIdx( uint16_t *origIdx );

    /**
     * TODO(kaehler): Comments
     */
    void setSampFreq( float sampFreq );

    static std::string channelToString( const ChannelT& data, size_t maxSamples );

    static std::string dataToString( const DataT& data, size_t maxChannels, size_t maxSamples );

protected:
    /**
     * name of the measurement device
     */
    std::string m_measurementDeviceName;

    /**
     * channel number in the measurement device defined by m_measurementDeviceName doxygen \ref
     * has one index for channel
     */
    uint16_t *m_origIdx;

    /**
     * power line frequency
     */
    float m_lineFreq;

    WLArrayList< std::string >::SPtr m_chanNames;

    /**
     * sampling frequency (unique within modality)
     */
    float m_sampFreq;

    /**
     * real world unit of the modality
     */
    LaBP::WEUnit::Enum m_chanUnit;

    /**
     * data is in unit m_chanUnit * 10^m_chanUnitExp
     */
    LaBP::WEExponent::Enum m_chanUnitExp;

    /**
     * cutoff frequency of highpass filter in analog processing chain
     */
    float m_analogHighPass;

    /**
     * cutoff frequency of lowpass filter in analog processing chain
     */
    float m_analogLowPass;

    /**
     * type of coordinate system used for m_chanPositions
     */
    LaBP::WECoordSystemName::Enum m_CoordSystem;

    /**
     * size of data buffer per channel in samples
     */
    uint32_t m_dataBuffSizePerChan;

    /**
     * offset in samples of current data block to the start of measurement
     */
    uint32_t m_dataOffsetIdx;

    /**
     * Raw data of the measurement.
     */
    DataSPtr m_data;

    // TODO(fuchs): snr estimate or/and noise covariance matrix for source localisation
};

std::ostream& operator<<( std::ostream &strm, const WLEMData& obj );

#endif  // WLEMDATA_H
