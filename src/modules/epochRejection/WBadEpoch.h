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

#ifndef WBADEPOCH_H_
#define WBADEPOCH_H_

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"

class WBadEpoch
{
public:

    /**
     * Constructor
     */
    WBadEpoch(WLEMMeasurement::ConstSPtr);

    /**
     * Destructor
     */
    virtual ~WBadEpoch();

    /**
     * Returns the const pointer to the EMM object.
     */
    WLEMMeasurement::ConstSPtr getEMM();

private:

    /**
     * The EMM object to store.
     */
    WLEMMeasurement::ConstSPtr m_emm;
};

#endif /* WBADEPOCH_H_ */
