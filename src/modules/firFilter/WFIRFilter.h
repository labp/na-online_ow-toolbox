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

#ifndef WFIRFILTER_H
#define WFIRFILTER_H

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"

class WFIRFilter
{
    friend class WFIRFilterTest;
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WFIRFilter > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WFIRFilter > ConstSPtr;

    typedef WLEMData::ScalarT ScalarT;

    static const std::string CLASS;

    struct WEFilterType
    {
        enum Enum
        {
            LOWPASS, HIGHPASS, BANDPASS, BANDSTOP, UNKNOWN
        };

        static std::vector< Enum > values();

        static std::string name( Enum value );
    };

    struct WEWindowsType
    {
        enum Enum
        {
            HAMMING, RECTANGLE, BARLETT, BLACKMAN, HANNING, UNKNOWN
        };

        static std::vector< Enum > values();

        static std::string name( Enum value );
    };

    WFIRFilter();

    explicit WFIRFilter( const std::string& pathToFcf );

    WFIRFilter( WEFilterType::Enum filtertype, WEWindowsType::Enum windowtype, int order, ScalarT sFreq, ScalarT cFreq1,
                    ScalarT cFreq2 );

    virtual ~WFIRFilter();

    WLEMData::SPtr filter( const WLEMData::ConstSPtr emdIn );

    void doPostProcessing( WLEMMeasurement::SPtr emmOut, WLEMMeasurement::ConstSPtr emmIn );

    void setFilterType( WEFilterType::Enum value, bool redesign = false );
    void setWindowsType( WEWindowsType::Enum value, bool redesign = false );
    void setOrder( size_t value, bool redesign = false );
    void setSamplingFrequency( ScalarT value, bool redesign = false );
    void setCutOffFrequency1( ScalarT value, bool redesign = false );
    void setCutOffFrequency2( ScalarT value, bool redesign = false );
    void setCoefficients( std::vector< ScalarT > values );
    bool setCoefficients( const std::string& pathToFcf );

    std::vector< ScalarT > getCoefficients();

    void design();
    void design( WEFilterType::Enum filtertype, WEWindowsType::Enum windowtype, size_t order, ScalarT sFreq, ScalarT cFreq1,
                    ScalarT cFreq2 );

    void reset();

protected:
    virtual void filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prev ) = 0;

    std::vector< ScalarT > m_coeffitients;
    WEWindowsType::Enum m_window;
    WEFilterType::Enum m_type;
    ScalarT m_sFreq;
    ScalarT m_cFreq1;
    ScalarT m_cFreq2;
    size_t m_order;
    std::vector< ScalarT > m_allPass;

    const WLEMData::DataT& getPreviousData( WLEMData::ConstSPtr emd );
    void storePreviousData( WLEMData::ConstSPtr emd );

private:
    void designLowpass( std::vector< ScalarT >* pCoeff, size_t order, ScalarT cFreq1, ScalarT sFreq,
                    WEWindowsType::Enum windowtype );
    void designHighpass();
    void designBandpass();
    void designBandstop();

    void normalizeCoeff( std::vector< ScalarT >* pCoeff );

    std::map< LaBP::WEModalityType::Enum, WLEMData::DataT > m_prevData;
    WLEMMeasurement::EDataT m_prevEvents;
};

#endif  // WFIRFILTER_H
