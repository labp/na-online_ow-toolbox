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

#ifndef WFIRFILTERCUDA_H
#define WFIRFILTERCUDA_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"

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

    WFIRFilterCuda();

    WFIRFilterCuda( WFIRFilter::WEFilterType::Enum filtertype, WLWindowFunction::WLEWindow windowtype, int order,
                    ScalarT sFreq, ScalarT cFreq1, ScalarT cFreq2 );

    explicit WFIRFilterCuda( const std::string& pathToFcf );

    virtual ~WFIRFilterCuda();

protected:
    virtual bool filter( WLEMData::DataT& out, const WLEMData::DataT& in, const WLEMData::DataT& prev );

    /**
     * Prepare the data and calls the CUDA kernel for FIR filter.
     *
     * \param output A 2 dimensional output array, first dimension = channel, second = samples.
     * \param input A 2 dimensional input array, first dimension = channel, second = samples.
     * \param previous Last samples of previous packet, 2 dimensional.
     * \param channelsNumber of channels.
     * \param samples Number of samples per channel.
     * \param coeffs Coefficient vector.
     * \param coeffSize Coefficient vector size.
     *
     * \return Elapsed time in ms.
     *
     * \throws WException
     * \throws WLBadAllocException
     */
    float cudaFilter( WLEMData::ScalarT* const output, const WLEMData::ScalarT* const input,
                    const WLEMData::ScalarT* const previous, size_t channels, size_t samples,
                    const WLEMData::ScalarT* const coeffs, size_t coeffSize );
};

#endif  // WFIRFILTERCUDA_H
