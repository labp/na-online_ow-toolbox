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

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"

#include "core/util/WLTimeProfiler.h"

#include "WFIRFilter.h"
#include "WFIRDesignWindow.h"

const std::string WFIRFilter::CLASS = "WFIRFilter";

WFIRFilter::WFIRFilter( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                double sFreq, double cFreq1, double cFreq2 )
{
    m_coeffitients = std::vector< double >();
    m_allPass = std::vector< double >();

    design( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 );
}

WFIRFilter::WFIRFilter( const char *pathToFcf )
{
    m_coeffitients = std::vector< double >();
    m_allPass = std::vector< double >();
    setCoefficients( pathToFcf );
}

WFIRFilter::~WFIRFilter()
{
}

WLEMData::SPtr WFIRFilter::filter( const WLEMData::ConstSPtr emdIn, LaBP::WLTimeProfiler::SPtr profiler )
{
    LaBP::WLTimeProfiler::SPtr emdProfiler( new LaBP::WLTimeProfiler( CLASS, "filter_emd" ) );
    emdProfiler->start();
    WLEMData::DataT in = emdIn->getData();
    boost::shared_ptr< WLEMData::DataT > out( new WLEMData::DataT() );
    out->reserve( in.size() );

    const WLEMData::DataT& prevData = getPreviousData( emdIn );
    filter( *out, in, prevData, emdProfiler );
    storePreviousData( emdIn );

    WLEMData::SPtr emdOut = emdIn->clone();
    emdOut->setData( out );

    emdProfiler->stopAndLog();
    if( profiler )
    {
        profiler->addChild( emdProfiler );
    }
    return emdOut;
}

void WFIRFilter::doPostProcessing( WLEMMeasurement::SPtr emmOut, WLEMMeasurement::ConstSPtr emmIn,
                LaBP::WLTimeProfiler::SPtr profiler )
{
    LaBP::WLTimeProfiler::SPtr emmProfiler( new LaBP::WLTimeProfiler( CLASS, "doPostProcess" ) );
    emmProfiler->start();

    boost::shared_ptr< WLEMMeasurement::EDataT > eventsIn = emmIn->getEventChannels();
    if( !eventsIn || eventsIn->empty() )
    {
        return;
    }

    boost::shared_ptr< WLEMMeasurement::EDataT > eventsOut( new WLEMMeasurement::EDataT() );
    const size_t channels = eventsIn->size();
    const size_t samples = eventsIn->front().size();
    // TODO(pieloth): correct prevSize / shift?
    const size_t prevSamples = m_coeffitients.size() / 2;

    // Prepare null events of "previous packet"
    if( m_prevEvents.empty() )
    {
        m_prevEvents.resize( channels );
        for( size_t chan = 0; chan < channels; ++chan )
        {
            m_prevEvents.at( chan ).resize( prevSamples, 0 );
        }
    }

    eventsOut->resize( channels );
    for( size_t chan = 0; chan < channels; ++chan )
    {
        eventsOut->at( chan ).reserve( samples );

        // Copy events of previous packet
        ( *eventsOut )[chan].insert( ( *eventsOut )[chan].end(), m_prevEvents.at( chan ).begin(), m_prevEvents.at( chan ).end() );

        // Copy events of current packet
        ( *eventsOut )[chan].insert( ( *eventsOut )[chan].end(), eventsIn->at( chan ).begin(),
                        eventsIn->at( chan ).end() - prevSamples );

        // Save events of current packet to previous events
        WAssert( samples >= prevSamples, "doPostProcess: samples >= prevSamples" );
        m_prevEvents[chan].assign( eventsIn->at( chan ).end() - prevSamples, eventsIn->at( chan ).end() );
        WAssertDebug( m_prevEvents[chan].size() == prevSamples, "doPostProcess: eventsOut[chan].size() == prevSamples" );
    }

    emmOut->setEventChannels( eventsOut );

    emmProfiler->stopAndLog();
    if( profiler )
    {
        profiler->addChild( emmProfiler );
    }
}

void WFIRFilter::setFilterType( WFIRFilter::WEFilterType::Enum value, bool redesign )
{
    m_type = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setWindowsType( WFIRFilter::WEWindowsType::Enum value, bool redesign )
{
    m_window = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setOrder( size_t value, bool redesign )
{
    m_order = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setSamplingFrequency( double value, bool redesign )
{
    m_sFreq = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setCutOffFrequency1( double value, bool redesign )
{
    m_cFreq1 = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setCutOffFrequency2( double value, bool redesign )
{
    m_cFreq2 = value;
    if( redesign )
    {
        design();
    }
}

void WFIRFilter::setCoefficients( std::vector< double > values, bool redesign )
{
    m_coeffitients = values;
    if( redesign )
    {
        design();
    }
    else
    {
        wlog::info( CLASS ) << "setCoefficients() m_coeffitients: " << m_coeffitients.size();
#ifdef DEBUG
        for( size_t i = 0; i < m_coeffitients.size(); i++ )
        {
            wlog::debug( CLASS ) << m_coeffitients[i];
        }
#endif // DEBUG
    }
}

bool WFIRFilter::setCoefficients( const char *pathToFcf, bool redesign )
{
    std::ifstream f( pathToFcf );
    std::string s;

    // find "Length"
    while( strcmp( s.c_str(), "Length" ) != 0 && !f.eof() )
    {
        f >> s;
    }

    if( f.eof() )
    {
        wlog::error( CLASS ) << "setCoefficients(): No coefficients found.";
        return false;
    }

    f >> s; // jump over the ":"
    f >> s; // now s contains the coefficientvektorlength

    int l;
    l = atoi( s.c_str() ); // read that length

    // find "Numerator:"
    while( strcmp( s.c_str(), "Numerator:" ) != 0 && !f.eof() )
    {
        f >> s;
    }

    if( f.eof() )
    {
        wlog::error( CLASS ) << "setCoefficients(): No coefficients found.";
        return false;
    }

    f >> s; // now s contains first coefficient

    m_coeffitients.clear();
    for( int i = 0; i < l; ++i )
    {
        m_coeffitients.push_back( atof( s.c_str() ) );
        f >> s;
    }

    f.close();
    if( redesign )
    {
        design();
    }
    else
    {
        wlog::info( CLASS ) << "setCoefficients() m_coeffitients: " << m_coeffitients.size();
#ifdef DEBUG
        for( size_t i = 0; i < m_coeffitients.size(); i++ )
        {
            wlog::debug( CLASS ) << m_coeffitients[i];
        }
#endif // DEBUG
    }

    return true;
}

std::vector< double > WFIRFilter::getCoefficients()
{
    return m_coeffitients;
}

void WFIRFilter::reset()
{
    m_prevData.clear();
    m_prevEvents.clear();
}

void WFIRFilter::design()
{
    this->reset();

    // prepare allPass for other filtertypes
    m_allPass.resize( m_order + 1, 0 );
    designLowpass( &m_allPass, m_order, 0.5, 1.0, m_window );

    wlog::info( CLASS ) << "design() m_allPass: " << m_allPass.size();
#ifdef DEBUG
    for( size_t i = 0; i < m_allPass.size(); i++ )
    {
        wlog::debug( CLASS ) << m_allPass[i];
    }
#endif // DEBUG
    m_coeffitients.resize( m_order + 1, 0 );

    switch( m_type )
    {
        case WFIRFilter::WEFilterType::LOWPASS:
            designLowpass( &m_coeffitients, m_order, m_cFreq1, m_sFreq, m_window );
            break;
        case WFIRFilter::WEFilterType::HIGHPASS:
            designHighpass();
            break;
        case WFIRFilter::WEFilterType::BANDPASS:
            designBandpass();
            break;
        case WFIRFilter::WEFilterType::BANDSTOP:
            designBandstop();
            break;
        default:
            WAssert( false, "Unknown WEFilterType!" );
            break;
    }

    wlog::info( CLASS ) << "design() m_coeffitients: " << m_coeffitients.size();
#ifdef DEBUG
    for( size_t i = 0; i < m_coeffitients.size(); i++ )
    {
        wlog::debug( CLASS ) << m_coeffitients[i];
    }
#endif // DEBUG
}

void WFIRFilter::design( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, size_t order,
                double sFreq, double cFreq1, double cFreq2 )
{
    m_type = filtertype;
    m_window = windowtype;
    m_order = order;
    m_sFreq = sFreq;
    m_cFreq1 = cFreq1;
    m_cFreq2 = cFreq2;

    design();
}

void WFIRFilter::designLowpass( std::vector< double >* pCoeff, size_t order, double cFreq, double sFreq,
                WFIRFilter::WEWindowsType::Enum windowtype )
{
    WAssert( pCoeff, "pCoeff is NULL!" );
    std::vector< double >& coeff = *pCoeff;
    WAssert( cFreq != 0, "cFreq != 0" );
    WAssert( sFreq != 0, "sFreq != 0" );
    WAssert( coeff.size() == order + 1, "coeff.size() == order + 1" );

    double a = 0.0;
    double b = 0.0;

    for( size_t i = 0; i < order + 1; ++i )
    {
        a = ( ( 2 * i - order ) * M_PI ) * ( cFreq / sFreq );
        b = WFIRDesignWindow::getFactor( windowtype, order, i );

        coeff[i] = ( ( sin( a ) / a ) * b );
    }

    if( order % 2 == 0 )
        coeff[order / 2] = 1;

    wlog::info( CLASS ) << "designLowpass() coeff: " << coeff.size();
#ifdef DEBUG
    for( size_t i = 0; i < coeff.size(); ++i )
    {
        wlog::debug( CLASS ) << coeff[i];
    }
#endif // DEBUG
    normalizeCoeff( &coeff );
}

void WFIRFilter::designHighpass( void )
{
    designLowpass( &m_coeffitients, m_order, m_cFreq1, m_sFreq, m_window );
    normalizeCoeff( &m_coeffitients );

    WAssert( m_allPass.size() == m_coeffitients.size(), "m_allPass.size() == m_coeffitients.size()" );
    for( size_t i = 0; i < m_order + 1; ++i )
    {
        m_coeffitients[i] = m_allPass[i] - m_coeffitients[i];
    }
}

void WFIRFilter::designBandpass( void )
{
    std::vector< double > tmpCoeff( m_coeffitients.size(), 0 );

    designLowpass( &m_coeffitients, m_order, m_cFreq1, m_sFreq, m_window );
    designLowpass( &tmpCoeff, m_order, m_cFreq2, m_sFreq, m_window );

    normalizeCoeff( &m_coeffitients );
    normalizeCoeff( &tmpCoeff );

    WAssert( tmpCoeff.size() == m_coeffitients.size(), "tmpCoeff.size() == m_coeffitients.size()" );
    for( size_t i = 0; i < m_coeffitients.size(); ++i )
    {
        // TODO(kaehler): noch klären wer von wem subtrahiert werden muss
        m_coeffitients[i] = tmpCoeff[i] - m_coeffitients[i];
    }
}

void WFIRFilter::designBandstop( void )
{
    designBandpass();

    WAssert( m_allPass.size() == m_coeffitients.size(), "m_allPass.size() == m_coeffitients.size()" );
    for( size_t i = 0; i < m_order + 1; i++ )
    {
        m_coeffitients[i] = m_allPass[i] - m_coeffitients[i];
    }
}

void WFIRFilter::normalizeCoeff( std::vector< double >* pCoeff )
{
    WAssert( pCoeff, "pCoeff is NULL!" );
    std::vector< double >& coeff = *pCoeff;
    double sum = 0;

    for( size_t i = 0; i < coeff.size(); ++i )
    {
        sum += coeff[i];
    }

    for( size_t i = 0; i < coeff.size(); i++ )
    {
        coeff[i] = coeff[i] / sum;
    }
}

std::vector< WFIRFilter::WEFilterType::Enum > WFIRFilter::WEFilterType::values()
{
    std::vector< WFIRFilter::WEFilterType::Enum > values;
    values.push_back( WEFilterType::BANDPASS );
    values.push_back( WEFilterType::BANDSTOP );
    values.push_back( WEFilterType::HIGHPASS );
    values.push_back( WEFilterType::LOWPASS );
    return values;
}

std::string WFIRFilter::WEFilterType::name( WFIRFilter::WEFilterType::Enum value )
{
    switch( value )
    {
        case WEFilterType::BANDPASS:
            return "Bandpass";
        case WEFilterType::BANDSTOP:
            return "Bandstop";
        case WEFilterType::HIGHPASS:
            return "Highpass";
        case WEFilterType::LOWPASS:
            return "Lowpass";
        default:
            WAssert( false, "Unknown WEFilterType!" );
            return LaBP::UNDEFINED;
    }
}

std::vector< WFIRFilter::WEWindowsType::Enum > WFIRFilter::WEWindowsType::values()
{
    std::vector< WFIRFilter::WEWindowsType::Enum > values;
    values.push_back( WEWindowsType::HAMMING );
    values.push_back( WEWindowsType::RECTANGLE );
    values.push_back( WEWindowsType::BARLETT );
    values.push_back( WEWindowsType::BLACKMAN );
    values.push_back( WEWindowsType::HANNING );
    return values;
}

std::string WFIRFilter::WEWindowsType::name( WFIRFilter::WEWindowsType::Enum value )
{
    switch( value )
    {
        case WEWindowsType::HAMMING:
            return "Hamming";
        case WEWindowsType::RECTANGLE:
            return "Rectangle";
        case WEWindowsType::BARLETT:
            return "Barlett";
        case WEWindowsType::BLACKMAN:
            return "Blackman";
        case WEWindowsType::HANNING:
            return "Hanning";
        default:
            WAssert( false, "Unknown WEWindowsType!" );
            return LaBP::UNDEFINED;
    }
}

const WLEMData::DataT& WFIRFilter::getPreviousData( WLEMData::ConstSPtr emd )
{
    if( m_prevData.count( emd->getModalityType() ) > 0 )
    {
        return m_prevData.find( emd->getModalityType() )->second;
    }
    else
    {
        wlog::debug( CLASS ) << "getPreviousData() generate zero data!";
        WLEMData::DataT data;
        data.resize( emd->getNrChans() );
        for( size_t i = 0; i < data.size(); ++i )
        {
            // TODO(pieloth): correct previous size / shift?
            data[i].resize( m_coeffitients.size(), 0 );
        }
        m_prevData[emd->getModalityType()] = data;
        return getPreviousData( emd );
    }
}

void WFIRFilter::storePreviousData( WLEMData::ConstSPtr emd )
{
    const WLEMData::DataT& dataIn = emd->getData();
    WAssert( m_coeffitients.size() <= emd->getSamplesPerChan(), "More coefficients than samples per channel!" );

    WLEMData::DataT data;
    data.resize( emd->getNrChans() );
    for( size_t i = 0; i < data.size(); ++i )
    {
        // TODO(pieloth): correct previous size / shift?
        data.reserve( m_coeffitients.size() );
        data[i].assign( dataIn[i].end() - m_coeffitients.size(), dataIn[i].end() );
        WAssertDebug( data[i].size() == m_coeffitients.size(), "storePreviousData: data[i].size() == m_coeffitients.size()" );
    }
    m_prevData[emd->getModalityType()] = data;
}
