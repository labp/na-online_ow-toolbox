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

#ifndef WFIRFILTER_H
#define WFIRFILTER_H

#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/preprocessing/WLWindowFunction.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"

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

        static std::set< Enum > values();

        static std::string name( Enum value );
    };

    WFIRFilter();

    explicit WFIRFilter( const std::string& pathToFcf );

    WFIRFilter( WEFilterType::Enum filtertype, WLWindowFunction::WLEWindow windowtype, int order, WLFreqT sFreq, WLFreqT cFreq1,
                    WLFreqT cFreq2 );

    virtual ~WFIRFilter();

    /**
     * Filters the data.
     *
     * \param emdIn
     * \return Filtered data
     * \throws WException
     */
    WLEMData::SPtr filter( const WLEMData::ConstSPtr emdIn );

    void doPostProcessing( WLEMMeasurement::SPtr emmOut, WLEMMeasurement::ConstSPtr emmIn );

    void setFilterType( WEFilterType::Enum value, bool redesign = false );
    void setWindowType( WLWindowFunction::WLEWindow value, bool redesign = false );
    void setOrder( size_t value, bool redesign = false );
    void setSamplingFrequency( WLFreqT value, bool redesign = false );
    void setCutOffFrequency1( WLFreqT value, bool redesign = false );
    void setCutOffFrequency2( WLFreqT value, bool redesign = false );
    void setCoefficients( std::vector< ScalarT > values );
    bool setCoefficients( const std::string& pathToFcf );

    std::vector< ScalarT > getCoefficients();

    void design();
    void design( WEFilterType::Enum filtertype, WLWindowFunction::WLEWindow windowtype, size_t order, WLFreqT sFreq,
                    WLFreqT cFreq1, WLFreqT cFreq2 );

    void reset();

protected:
    virtual bool filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prev ) = 0;

    std::vector< ScalarT > m_coeffitients;
    WLWindowFunction::WLEWindow m_window;
    WEFilterType::Enum m_type;
    WLFreqT m_sFreq;
    WLFreqT m_cFreq1;
    WLFreqT m_cFreq2;
    size_t m_order;
    std::vector< ScalarT > m_allPass;

    const WLEMData::DataT& getPreviousData( WLEMData::ConstSPtr emd );
    void storePreviousData( WLEMData::ConstSPtr emd );

private:
    void designLowpass( std::vector< ScalarT >* pCoeff, size_t order, WLFreqT cFreq1, WLFreqT sFreq,
                    WLWindowFunction::WLEWindow windowtype );
    void designHighpass();
    void designBandpass();
    void designBandstop();

    void normalizeCoeff( std::vector< ScalarT >* pCoeff );

    std::map< WLEModality::Enum, WLEMData::DataT > m_prevData;
    WLEMMeasurement::EDataT m_prevEvents;
};

#endif  // WFIRFILTER_H
