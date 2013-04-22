//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS, Copyright 2010 RRZK University of Cologne
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

#ifndef WFIRFILTERCUDA_H
#define WFIRFILTERCUDA_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "WFIRFilter.h"

class WFIRFilterCuda: public WFIRFilter
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WFIRFilterCuda > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WFIRFilterCuda > ConstSPtr;

    static const std::string CLASS;

    WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order,
                    double sFreq, double cFreq1, double cFreq2 );
    explicit WFIRFilterCuda( const char *pathToFcf );

protected:
    void filter( LaBP::WDataSetEMMEMD::DataT& out, const LaBP::WDataSetEMMEMD::DataT& in, const LaBP::WDataSetEMMEMD::DataT& prev,
                    LaBP::WLTimeProfiler::SPtr profiler );
};

#endif  // WFIRFILTERCUDA_H
