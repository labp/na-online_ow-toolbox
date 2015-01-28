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

#ifndef WEPOCHAVERAGINGMOVING_H
#define WEPOCHAVERAGINGMOVING_H

#include <cstddef>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"

#include "WEpochAveraging.h"

/**
 * Class for the calculation of moving average. The average is calculated from the k-1 previous and current object.
 *
 * \author  Christof Pieloth
 */
class WEpochAveragingMoving: public WEpochAveraging
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochAveragingMoving > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WEpochAveragingMoving > ConstSPtr;

    WEpochAveragingMoving( WLSampleNrT tbase, size_t size );

    virtual ~WEpochAveragingMoving();

    WLEMMeasurement::SPtr getAverage( const WLEMMeasurement::ConstSPtr emm );
    void reset();

    size_t getCount() const;

    /**
     * Returns the number of objects to be calculated
     */
    size_t getSize() const;

    /**
     * Set the number of objects to be calculated. The averager can be reset!
     */
    void setSize( size_t m_size );

private:
    /**
     * Count of WDataSetEMM objects.
     */
    size_t m_size;

    /**
     * Adds a WDataSetEMM object to the buffer.
     */
    void pushBuffer( const WLEMMeasurement::ConstSPtr emm );

    /**
     * Circular, overriding buffer. Holds the last k WDataSetEMM objects to calculate the average of the last k objects.
     */
    std::vector< WLEMMeasurement::ConstSPtr > m_buffer;

    /**
     * Index for buffer.
     */
    size_t m_ptr;
};

#endif  // WEPOCHAVERAGINGMOVING_H
