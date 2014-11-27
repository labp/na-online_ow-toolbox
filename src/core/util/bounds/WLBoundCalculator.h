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

#ifndef WLBOUNDCALCULATOR_H_
#define WLBOUNDCALCULATOR_H_

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "WLABoundCalculator.h"

/**
 * WLBoundCalculator implements the WLABoundCalculator interface to realize a bound calculation algorithm.
 *
 * \author maschke
 * \ingroup util
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
     * \param alpha A constant used by the algorithm.
     */
    explicit WLBoundCalculator( WLEMData::ScalarT alpha = 1.5 );

    /**
     * Destroys the WLBoundCalculator.
     */
    virtual ~WLBoundCalculator();

    /**
     * Calculates the maximum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * \param data The data matrix.
     * \return Returns the calculated maximum.
     */
    WLEMData::ScalarT getMax( const WLEMData::DataT& data );

    /**
     * Calculates the minimum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * \param data The data matrix.
     * \return Returns the calculated minimum.
     */
    WLEMData::ScalarT getMin( const WLEMData::DataT& data );

private:
    /**
     * A constant for calculation.
     */
    WLEMData::ScalarT m_alpha;
};

#endif  // WLBOUNDCALCULATOR_H_
