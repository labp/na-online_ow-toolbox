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

WFIRFilterCpu::WFIRFilterCpu( WFIRFilter::WEFilterType::Enum filtertype, WLWindowsFunction::WLEWindows windowtype, int order,
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
    const WLSampleNrT nbCoeff = m_coeffitients.size();
    out.setZero();
    for( WLSampleIdxT n = 0; n < in.cols(); ++n )
    {
        for( WLSampleIdxT k = 0; k < nbCoeff; ++k )
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
