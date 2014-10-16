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

#ifndef WEPOCHREJECTIONSINGLE_H_
#define WEPOCHREJECTIONSINGLE_H_

#include <string>

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
     * Method to reset the process parameter.
     */
    void initRejection();
};

#endif  // WEPOCHREJECTIONSINGLE_H_
