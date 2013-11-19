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

#ifndef WEPOCHAVERAGING_H
#define WEPOCHAVERAGING_H

#include <cstddef>
#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"

/**
 * Abstract class for all average calculations on WDataSetEMM.
 *
 * \author  Christof Pieloth
 */
class WEpochAveraging: public boost::enable_shared_from_this< WEpochAveraging >
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochAveraging > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WEpochAveraging > ConstSPtr;

    explicit WEpochAveraging( size_t tbase );

    virtual ~WEpochAveraging();

    /**
     * Returns the number of WDataSetEMM objects, which are used for the calculation.
     */
    virtual size_t getCount() const;

    size_t getTBase() const;

    void setTBase( size_t tbase, bool reset = true );

    /**
     * Calculates the average of the passed in and the previously passed in WDataSetEMM objects.
     *
     * \return  A new WDataSetEMM object, which holds the averaged data. Attributes are cloned from passed in object.
     */
    virtual WLEMMeasurement::SPtr getAverage( const WLEMMeasurement::ConstSPtr emm ) = 0;

    /**
     * Resets all necessary attributes and objects to start a new calculation.
     */
    virtual void reset();

    /**
     * Cast to Average if possible.
     *
     * @return Shared Pointer< AVG >
     */
    template< typename AVG >
    boost::shared_ptr< AVG > getAs()
    {
        return boost::dynamic_pointer_cast< AVG >( shared_from_this() );
    }

    /**
     * Cast to Average if possible.
     *
     * @return Shared Pointer< const AVG >
     */
    template< typename AVG >
    boost::shared_ptr< const AVG > getAs() const
    {
        return boost::dynamic_pointer_cast< AVG >( shared_from_this() );
    }

protected:
    /**
     * Counter for the passed in WDataSetEMM objects, which shall be used for the average.
     */
    size_t m_count;

    size_t m_tbase;

    /**
     * Does a baseline correction for each channel using m_tbase samples for mean.
     *
     * \return new EMM instance with corrected data.
     */
    WLEMMeasurement::SPtr baseline( WLEMMeasurement::ConstSPtr emm );
};

#endif  // WEPOCHAVERAGING_H
