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

#ifndef WEPOCHREJECTIONTOTAL_H_
#define WEPOCHREJECTIONTOTAL_H_

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"

#include "WEpochRejection.h"

class WEpochRejectionTotal: public WEpochRejection
{
public:

    static const std::string CLASS;

    /**
     * A shared pointer on the class.
     */
    typedef boost::shared_ptr< WEpochRejectionTotal > SPtr;

    /**
     * Constructor
     */
    WEpochRejectionTotal();

    /**
     * Destructor
     */
    ~WEpochRejectionTotal();

    /**
     * Proceeds the rejection of the all modalities for the given input based on the
     * user defined level values.
     *
     * \return A boolean value, which specifies, whether or not the input object has to reject.
     */
    bool doRejection( const WLEMMeasurement::ConstSPtr emm );

private:

    /**
     * Calculates the difference from peek to peek for the given matrix and compares with the threshold.
     *
     * @param data The matrix.
     * @param threshold The threshold.
     * @return Returns true if the difference was larger than the threshold, else false.
     */
    bool calcRejection( const WLEMData::DataT& data, double threshold );

    /**
     * Calculates the difference from peek to peek for the given matrix and compares with the threshold.
     *
     * @param data The matrix.
     * @param threshold The threshold.
     * @return Returns true if the difference was larger than the threshold, else false.
     */
    bool calcRejection( const WLEMDMEG::DataSPtr data, double threshold );

};

#endif /* WEPOCHREJECTIONTOTAL_H_ */
