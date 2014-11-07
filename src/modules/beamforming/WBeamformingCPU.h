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

#ifndef WBEAMFORMINGCPU_H_
#define WBEAMFORMINGCPU_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "WBeamforming.h"

/**
 * \author ehrlich
 */
class WBeamformingCPU: public WBeamforming
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WBeamformingCPU > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WBeamformingCPU > ConstSPtr;

    static const std::string CLASS;

    WBeamformingCPU();
    virtual ~WBeamformingCPU();

    WLEMDSource::SPtr beam( WLEMData::ConstSPtr emd );
};

#endif  // WBEAMFORMINGCPU_H_
