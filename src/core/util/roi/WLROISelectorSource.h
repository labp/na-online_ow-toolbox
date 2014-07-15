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

#ifndef WLROISELECTORSOURCE_H_
#define WLROISELECTORSOURCE_H_

#include <vector>

#include "core/data/emd/WLEMData.h"
#include "WLROISelector.h"

/**
 * The WLROISelectorSource is a derivation of the abstract WLROISelector.
 * It provides the adapter between the ROI configuration, the ROI Manager of OpenWalnut
 * and the source reconstruction algorithm.
 */
class WLROISelectorSource: public WLROISelector< WLEMData, std::vector< size_t > >
{

public:

    /**
     * Constructs a new WLROISelectorSource.
     *
     * @param data The data container.
     */
    WLROISelectorSource( WLEMData::SPtr data );

protected:

    /**
     * Recalculates the filter structure to select the channels includes by the ROI.
     */
    void recalculate();

};

#endif /* WLROISELECTORSOURCE_H_ */
