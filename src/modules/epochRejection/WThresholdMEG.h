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

#ifndef WTHRESHOLDMEG_H_
#define WTHRESHOLDMEG_H_

#include "core/data/WLEMMEnumTypes.h"

#include "WThreshold.h"

using namespace LaBP;

/**
 * This class is a subclass of Threshold to specialize the MEG modality for their special channels.
 */
class WThresholdMEG: public WThreshold
{
public:

    /**
     * Constructor to create the threshold object.
     *
     * @param modalityType The modality type.
     * @param coilType The coil type.
     * @param value The threshold value.
     */
    WThresholdMEG( LaBP::WEGeneralCoilType::Enum coilType, double value );

    /**
     * Gets the coil type.
     *
     * @return The coil type.
     */
    LaBP::WEGeneralCoilType::Enum getCoilType() const;

protected:

    /**
     * The coil type.
     */
    LaBP::WEGeneralCoilType::Enum m_coilType;
};

#endif /* WTHRESHOLDMEG_H_ */
