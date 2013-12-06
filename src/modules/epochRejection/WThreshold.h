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

#ifndef WTHRESHOLD_H_
#define WTHRESHOLD_H_

#include "core/data/WLEMMEnumTypes.h"

using namespace LaBP;

/**
 * The class represents a threshold object specified by a modality type and a value.
 */
class WThreshold
{
public:

    /**
     * Constructs the new threshold object
     *
     * @param modalityType The modality.
     * @param value The threshold.
     */
    WThreshold(LaBP::WEModalityType::Enum modalityType, double value);

    /**
     * Gets the threshold value.
     *
     * @return The threshold.
     */
    double getValue() const;

    /**
     * Gets the modality type for the thresholds usage.
     *
     * @return The modality type.
     */
    LaBP::WEModalityType::Enum getModaliyType() const;

protected:

    /**
     * Threshold value.
     */
    double m_value;

    /**
     * Threshold modality type.
     */
    LaBP::WEModalityType::Enum m_modalityType;
};

#endif /* WTHRESHOLD_H_ */
