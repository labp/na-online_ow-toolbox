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

#ifndef WEPOCHREJECTIONSINGLE_H_
#define WEPOCHREJECTIONSINGLE_H_

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"

#include "WEpochRejection.h"

class WEpochRejectionSingle: public WEpochRejection
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochRejectionSingle > SPtr;

    /**
     * Constructor
     */
    WEpochRejectionSingle();

    /**
     * Destructor
     */
    ~WEpochRejectionSingle();

    /**
     * Proceeds the rejection of the all modalities for the given input based on the
     * user defined level values.
     *
     * \return A boolean value, which specifies, whether or not the input object has to reject.
     */
    bool doRejection( const WLEMMeasurement::ConstSPtr emm );

    /**
     * Defines whether or not a new channel was added to the bad channel collection during the rejection process.
     *
     * @return True if a new bad channel was detected, else false.
     */
    bool isBadChannelUpdated() const;

protected:

    /**
     * Flag to define whether or not a new bad channel was found and added to the bad channel manager.
     * By default the flag is set to 'false'.
     */
    bool m_BadChannelUpdated;
};

#endif /* WEPOCHREJECTIONSINGLE_H_ */
