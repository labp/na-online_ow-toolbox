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

#ifndef WBADEPOCHMANAGER_H_
#define WBADEPOCHMANAGER_H_

#include "boost/circular_buffer.hpp"
#include "boost/shared_ptr.hpp"

#include "core/data/WLEMMeasurement.h"

class WBadEpochManager
{
public:
    typedef boost::circular_buffer< WLEMMeasurement::SPtr > CircBuff;

    /**
     * Definition of a shared pointer on a circular buffer.
     */
    typedef boost::shared_ptr< CircBuff > CircBuffSPtr;

    /**
     * Static method to return the singleton instance.
     *
     * @return Returns the current class as singleton.
     */
    static WBadEpochManager *instance();

    /**
     * Gets a reference on the circular buffer.
     *
     * @return A reference on the circular buffer.
     */
    CircBuffSPtr getBuffer();

    size_t getBufferSize();

    /**
     * Sets the shared pointer of the circular buffer.
     *
     * @param buffer
     */
    void setBuffer( CircBuffSPtr buffer );

    void resizeBuffer( size_t size );

    void reset();

private:
    /**
     * The private constructor initialize the buffer with a size of 5 elements.
     * If the buffer has to be resized, you have to use the boost::circular_buffer's rset_capacity() method.
     */
    WBadEpochManager();
    WBadEpochManager( const WBadEpochManager& );
    virtual ~WBadEpochManager();

    /**
     * A shared pointer on a circular buffer, containing BadEpoch objects.
     */
    CircBuffSPtr m_buffer;

    static WBadEpochManager *m_instance;
};

#endif  // WBADEPOCHMANAGER_H_
