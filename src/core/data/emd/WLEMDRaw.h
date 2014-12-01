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

#ifndef WLEMDRAW_H_
#define WLEMDRAW_H_

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include "WLEMData.h"

/**
 * Type for raw or generic EMData.
 * This can be used for temporary variables or for low-level I/O, e.g. acquisition clients.
 *
 * \author pieloth
 * \ingroup data
 */
class WLEMDRaw: public WLEMData
{
public:
    typedef boost::shared_ptr< WLEMDRaw > SPtr;
    typedef boost::shared_ptr< const WLEMDRaw > ConstSPtr;

    typedef Eigen::RowVectorXi ChanPicksT;

    WLEMDRaw();
    virtual ~WLEMDRaw();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    using WLEMData::getData;

    /**
     * Picks the requested channels from the data.
     *
     * \param picks Channel Indices to pick
     * \param checkIndices If true, checks if each index relates to a channel. Default: true
     * \return A new instance of data
     * \throws WOutOfBounds If number picks or indices are smaller/greater than the number of channels.
     */
    DataSPtr getData( const ChanPicksT& picks, bool checkIndices = true ) const;
};

#endif  // WLEMDRAW_H_
