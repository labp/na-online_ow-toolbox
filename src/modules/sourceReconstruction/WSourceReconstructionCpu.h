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

#ifndef WSOURCERECONSTRUCTIONCPU_H_
#define WSOURCERECONSTRUCTIONCPU_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "core/util/WLTimeProfiler.h"

#include "WSourceReconstruction.h"

class WSourceReconstructionCpu: public WSourceReconstruction
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WSourceReconstructionCpu > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WSourceReconstructionCpu > ConstSPtr;

    static const std::string CLASS;

    WSourceReconstructionCpu();
    virtual ~WSourceReconstructionCpu();

    // Move up to base class virtual bool calculateInverseSolution( const MatrixT& noiseCov, const MatrixT& dataCov );

    virtual WLEMDSource::SPtr reconstruct( WLEMData::ConstSPtr emd, LaBP::WLTimeProfiler::SPtr profiler );
};

#endif  // WSOURCERECONSTRUCTIONCPU_H_
