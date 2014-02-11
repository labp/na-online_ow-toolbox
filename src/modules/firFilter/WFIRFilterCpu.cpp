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

#include "core/data/emd/WLEMData.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WFIRFilter.h"
#include "WFIRFilterCpu.h"

const std::string WFIRFilterCpu::CLASS = "WFIRFilterCpu";

WFIRFilterCpu::WFIRFilterCpu() :
                WFIRFilter()
{
}

WFIRFilterCpu::WFIRFilterCpu( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                ScalarT sFreq, ScalarT cFreq1, ScalarT cFreq2 ) :
                WFIRFilter( filtertype, windowtype, order, sFreq, cFreq1, cFreq2 )
{
}

WFIRFilterCpu::WFIRFilterCpu( const std::string& pathToFcf ) :
                WFIRFilter( pathToFcf )
{
}

WFIRFilterCpu::~WFIRFilterCpu()
{
}

bool WFIRFilterCpu::filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prevData )
{
    wlog::debug( CLASS ) << "filter() called!";
    WLTimeProfiler prfTime( CLASS, "filter" );

    // CHANGE original: for( int n = 1; (uint) n < in.size(); n++ )
    const size_t nbCoeff = m_coeffitients.size();
    out.setZero();
    for( WLEMData::ChannelT::Index n = 0; n < in.cols(); ++n )
    {
        for( size_t k = 0; k < nbCoeff; ++k )
        {
            // CHANGE from ( long int )( n - k ) >= 0 ? m_coeffitients[k] * in[n - k] : 0;
            // tmp += ( n >= k ) ? m_coeffitients[k] * in[n - k] : 0;
            if( n >= k )
            {
                out.col( n ) += m_coeffitients[k] * in.col( n - k );
            }
            else
            {
                out.col( n ) += m_coeffitients[k] * prevData.col( nbCoeff - ( k - n ) );
            }
        }
    }

    return true;
}
