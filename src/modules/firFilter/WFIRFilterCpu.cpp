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

#include <cstddef>
#include <string>

#include <core/common/WLogger.h>

#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/util/WLTimeProfiler.h"

#include "WFIRFilter.h"
#include "WFIRFilterCpu.h"

const std::string WFIRFilterCpu::CLASS = "WFIRFilterCpu";

WFIRFilterCpu::WFIRFilterCpu( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                double sFreq, double cFreq1, double cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCpu::WFIRFilterCpu( const char *pathToFcf ) :
                WFIRFilter( pathToFcf )
{
}

void WFIRFilterCpu::filter( LaBP::WDataSetEMMEMD::DataT& out, const LaBP::WDataSetEMMEMD::DataT& in,
                const LaBP::WDataSetEMMEMD::DataT& prevData, LaBP::WLTimeProfiler::SPtr profiler )
{
    wlog::debug( CLASS ) << "filter() called!";
    LaBP::WLTimeProfiler::SPtr emdProfiler( new LaBP::WLTimeProfiler( CLASS, "filter_data" ) );
    emdProfiler->start();

    for( size_t i = 0; i < in.size(); ++i )
    {
        LaBP::WDataSetEMMEMD::ChannelT outChan; // generate a new dimension for every channel
        outChan.reserve( in[i].size() );
        out.push_back( outChan );

        filterSingleChannel( out[i], in[i], prevData[i] );
    }

    emdProfiler->stopAndLog();
    if( profiler )
    {
        profiler->addChild( emdProfiler );
    }
}

void WFIRFilterCpu::filterSingleChannel( LaBP::WDataSetEMMEMD::ChannelT& out, const LaBP::WDataSetEMMEMD::ChannelT& in,
                const LaBP::WDataSetEMMEMD::ChannelT& prev )
{
    // TODO(pieloth): check
    LaBP::WDataSetEMMEMD::SampleT tmp = 0.0;

    // CHANGE original: for( int n = 1; (uint) n < in.size(); n++ )
    const size_t nbCoeff = m_coeffitients.size();
    for( size_t n = 0; n < in.size(); ++n )
    {
        tmp = 0.0;
        for( size_t k = 0; k < nbCoeff; ++k )
        {
            // CHANGE from ( long int )( n - k ) >= 0 ? m_coeffitients[k] * in[n - k] : 0;
            //tmp += ( n >= k ) ? m_coeffitients[k] * in[n - k] : 0;
            if( n >= k )
            {
                tmp += m_coeffitients[k] * in[n - k];
            }
            else
            {
                tmp += m_coeffitients[k] * prev[nbCoeff - ( k - n )];
            }
        }
        out.push_back( tmp );
    }
}
