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

#include "core/data/emd/WLEMData.h"

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

    WFIRFilterCpu();

    WFIRFilterCpu( WFIRFilter::WEFilterType::Enum filtertype, WFIRFilter::WEWindowsType::Enum windowtype, int order, ScalarT sFreq,
                    ScalarT cFreq1, ScalarT cFreq2 );

    explicit WFIRFilterCpu( const std::string& pathToFcf );

    virtual ~WFIRFilterCpu();

protected:
    virtual bool filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prevData );

};

#endif  // WFIRFILTERCPU_H
