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

#ifndef WLBOUNDCALCULATOR_H_
#define WLBOUNDCALCULATOR_H_

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "WLABoundCalculator.h"

/**
 * WLBoundCalculator implements the WLIBoundCalculator interface to realize a bound calculation algorithm.
 */
class WLBoundCalculator: public WLABoundCalculator
{
public:

    /**
     * A shared pointer on a WLBoundCalculator.
     */
    typedef boost::shared_ptr< WLBoundCalculator > SPtr;

    /**
     * A shared pointer on a constant WLBoundCalculator.
     */
    typedef boost::shared_ptr< const WLBoundCalculator > ConstSPtr;

    /**
     * Constructs a new WLBoundCalculator.
     *
     * @param alpha A constant used by the algorithm.
     */
    explicit WLBoundCalculator( WLEMData::ScalarT alpha = 1.5 );

    /**
     * Destroys the WLBoundCalculator.
     */
    virtual ~WLBoundCalculator();

    /**
     * Calculates the bound for the 3D view.
     *
     * @param emm The measurement object.
     * @param modality The modality to display.
     * @return Returns a vector, which contains the calculated bounds.
     */
    WLArrayList< WLEMData::ScalarT > getBounds3D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality );

    /**
     * Calculates the maximum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * @param data The data matrix.
     * @return Returns the calculated maximum.
     */
    WLEMData::ScalarT getMax( const WLEMData::DataT& data );

    /**
     * Calculates the minimum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * @param data The data matrix.
     * @return Returns the calculated minimum.
     */
    WLEMData::ScalarT getMin( const WLEMData::DataT& data );

private:

    /**
     * A constant for calculation.
     */
    WLEMData::ScalarT m_alpha;
};

#endif  // WLBOUNDCALCULATOR_H_
