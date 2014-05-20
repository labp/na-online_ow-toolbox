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

#ifndef WLBOUNDCALCULATORHISTOGRAM_H_
#define WLBOUNDCALCULATORHISTOGRAM_H_

#include "WLABoundCalculator.h"

/**
 * WLBoundCalculatorHistogram implements the WLIBoundCalculator interface to realize a bound calculation algorithm.
 * The class uses a histogram to determine, which of the data elements have to include into the bounds and which of
 * them could be excluded.
 * The calculation uses the absoulte values of the matrix to determine the minimum.
 */
class WLBoundCalculatorHistogram: public WLABoundCalculator
{

public:

    /**
     * A shared pointer on a WLBoundCalculatorHistogram.
     */
    typedef boost::shared_ptr< WLBoundCalculatorHistogram > SPtr;

    /**
     * A shared pointer on a constant WLBoundCalculatorHistogram.
     */
    typedef boost::shared_ptr< const WLBoundCalculatorHistogram > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WLBoundCalculatorHistogram.
     */
    WLBoundCalculatorHistogram();

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

    /**
     * Gets the percental value.
     *
     * @return Returns a double value.
     */
    double getPercent();

    /**
     * Sets the percental value.
     *
     * @param percent The new value.
     */
    void setPercent( double percent );

private:

    /**
     * the percental limit for the algorithim.
     */
    double m_percent;
};

#endif /* WLBOUNDCALCULATORHISTOGRAM_H_ */
