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

#ifndef WLABOUNDCALCULATOR_H_
#define WLABOUNDCALCULATOR_H_

#include <string>
#include <utility> // std::pair

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/data/WLEMMeasurement.h"

/**
 * WLABoundCalculator is the base interface for calculating the boundaries in the modules 2D and 3D views.
 * There exists different algorithms for identifying the bounds, so the implementations are realized as a
 * parameterized strategy pattern.
 *
 * \author maschke
 */
class WLABoundCalculator: public boost::enable_shared_from_this< WLABoundCalculator >
{

public:

    /**
     * A shared pointer on a WLIBoundCalculator.
     */
    typedef boost::shared_ptr< WLABoundCalculator > SPtr;

    /**
     * A shared pointer on a constant WLIBoundCalculator.
     */
    typedef boost::shared_ptr< const WLABoundCalculator > ConstSPtr;

    /**
     * A pair of Min/Max. Minimum is stored in first and maximum in second.
     */
    typedef std::pair< WLEMData::ScalarT, WLEMData::ScalarT > MinMax;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Destroys the WLIBoundCalculator.
     */
    virtual ~WLABoundCalculator();

    /**
     * Calculates the bounds for a 2D view.
     *
     * \param emm The measurement object.
     * \param modality The modality to display.
     * \return Returns a pair, which contains the minimal (0) and maximal amplitude scaling.
     */
    virtual MinMax getBounds2D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality );

    /**
     * Calculates the bound for the 3D view.
     *
     * \param emm The measurement object.
     * \param modality The modality to display.
     * \return Returns a pair, which contains the calculated bounds.
     */
    virtual MinMax getBounds3D( WLEMMeasurement::ConstSPtr emm, WLEModality::Enum modality );

    /**
     * Calculates the maximum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * \param data The data matrix.
     * \return Returns the calculated maximum.
     */
    virtual WLEMData::ScalarT getMax( const WLEMData::DataT& data ) = 0;

    /**
     * Calculates the minimum.
     * The results of this methods depends on the algorithms, implemented by the derived classes. So they can vary.
     *
     * \param data The data matrix.
     * \return Returns the calculated minimum.
     */
    virtual WLEMData::ScalarT getMin( const WLEMData::DataT& data ) = 0;

    /**
     * Casts the base WLABoundCalculator to a derived class.
     *
     * \return Returns a shared pointer on the casted class.
     */
    template< typename T >
    boost::shared_ptr< T > getAs()
    {
        return boost::dynamic_pointer_cast< T >( shared_from_this() );
    }

    /**
     * Casts the base WLABoundCalculator to a derived class.
     *
     * \return Returns a shared pointer on the casted class.
     */
    template< typename T >
    boost::shared_ptr< const T > getAs() const
    {
        return boost::dynamic_pointer_cast< T >( shared_from_this() );
    }

};

#endif /* WLABOUNDCALCULATOR_H_ */
