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

#ifndef WFIRFILTERCPU_H
#define WFIRFILTERCPU_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMD.h"

#include "WFIRFilter.h"

class WFIRFilterCpu: public WFIRFilter
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WFIRFilterCpu > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WFIRFilterCpu > ConstSPtr;

    static const std::string CLASS;

    WFIRFilterCpu( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order, double sFreq,
                    double cFreq1, double cFreq2 );
    explicit WFIRFilterCpu( const char *pathToFcf );
    //virtual ~WMFIRFilterCpu();

protected:
    void filter( LaBP::WLEMD::DataT& out, const LaBP::WLEMD::DataT& in,
                    const LaBP::WLEMD::DataT& prevData, LaBP::WLTimeProfiler::SPtr profiler );

private:
    void filterSingleChannel( LaBP::WLEMD::ChannelT& out, const LaBP::WLEMD::ChannelT& in,
                    const LaBP::WLEMD::ChannelT& prev );
};

#endif  // WFIRFILTERCPU_H
