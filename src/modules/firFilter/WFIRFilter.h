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

#include "core/common/WLTimeProfiler.h"
#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEMD.h"

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

    static const std::string CLASS;

    struct WEFilterType
    {
        enum Enum
        {
            LOWPASS, HIGHPASS, BANDPASS, BANDSTOP
        };

        static std::vector< Enum > values();

        static std::string name( Enum value );
    };

    struct WEWindowsType
    {
        enum Enum
        {
            HAMMING, RECTANGLE, BARLETT, BLACKMAN, HANNING
        };

        static std::vector< Enum > values();

        static std::string name( Enum value );
    };

    WFIRFilter( WEFilterType::Enum filtertype, WEWindowsType::Enum windowtype, int order, double sFreq, double cFreq1,
                    double cFreq2 );
    explicit WFIRFilter( const char *pathToFcf );

    virtual ~WFIRFilter();

    LaBP::WDataSetEMMEMD::SPtr filter( const LaBP::WDataSetEMMEMD::ConstSPtr emdIn, LaBP::WLTimeProfiler::SPtr profiler );

    void doPostProcessing( LaBP::WDataSetEMM::SPtr emmOut, LaBP::WDataSetEMM::ConstSPtr emmIn,
                    LaBP::WLTimeProfiler::SPtr profiler );

    void setFilterType( WEFilterType::Enum value, bool redesign = false );
    void setWindowsType( WEWindowsType::Enum value, bool redesign = false );
    void setOrder( size_t value, bool redesign = false );
    void setSamplingFrequency( double value, bool redesign = false );
    void setCutOffFrequency1( double value, bool redesign = false );
    void setCutOffFrequency2( double value, bool redesign = false );
    void setCoefficients( std::vector< double > values, bool redesign = false );
    bool setCoefficients( const char *pathToFcf, bool redesign = false );

    std::vector< double > getCoefficients();

    void design();
    void design( WEFilterType::Enum filtertype, WEWindowsType::Enum windowtype, size_t order, double sFreq, double cFreq1,
                    double cFreq2 );

protected:
    virtual void filter( LaBP::WDataSetEMMEMD::DataT& out, const LaBP::WDataSetEMMEMD::DataT& in,
                    const LaBP::WDataSetEMMEMD::DataT& prev, LaBP::WLTimeProfiler::SPtr profiler ) = 0;

    std::vector< double > m_coeffitients;
    WEWindowsType::Enum m_window;
    WEFilterType::Enum m_type;
    double m_sFreq;
    double m_cFreq1;
    double m_cFreq2;
    size_t m_order;
    std::vector< double > m_allPass;

    const LaBP::WDataSetEMMEMD::DataT& getPreviousData( LaBP::WDataSetEMMEMD::ConstSPtr emd );
    void storePreviousData( LaBP::WDataSetEMMEMD::ConstSPtr emd );

private:
    void designLowpass( std::vector< double >* pCoeff, size_t order, double cFreq1, double sFreq, WEWindowsType::Enum windowtype );
    void designHighpass( void );
    void designBandpass( void );
    void designBandstop( void );

    void normalizeCoeff( std::vector< double >* pCoeff );

    std::map< LaBP::WEModalityType::Enum, LaBP::WDataSetEMMEMD::DataT > m_prevData;
    LaBP::WDataSetEMM::EDataT m_prevEvents;
};

#endif  // WFIRFILTER_H
