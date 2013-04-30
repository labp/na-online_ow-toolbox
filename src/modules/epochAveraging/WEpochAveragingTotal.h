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

#ifndef WEPOCHAVERAGINGTOTAL_H
#define WEPOCHAVERAGINGTOTAL_H

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLDataSetEMM.h"

#include "WEpochAveraging.h"

/**
 * Class for the calculation of total average. The average is calculated from all previous and current data.
 *
 * \author  Christof Pieloth
 */
class WEpochAveragingTotal: public WEpochAveraging
{
public:
    static const string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochAveragingTotal > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WEpochAveragingTotal > ConstSPtr;

    explicit WEpochAveragingTotal( size_t tbase );

    virtual ~WEpochAveragingTotal();

    LaBP::WLDataSetEMM::SPtr getAverage( const LaBP::WLDataSetEMM::ConstSPtr emm );

    void reset();

private:
    /**
     * Running sum for division.
     */
    LaBP::WLDataSetEMM::SPtr m_emmSum;

    /**
     * Sums the data of all modalities and channels to the corresponding data of the running sum.
     */
    void addEmmSum( const LaBP::WLDataSetEMM::ConstSPtr emm );

    /**
     * Creates the WDataSetEMM object for the running sum, if this is not initialized.
     * The modality count, channel count and samples size of the passed in object is used for all future data.
     */
    void checkEmmSum( const LaBP::WLDataSetEMM::ConstSPtr emm );
};

#endif  // WEPOCHAVERAGINGTOTAL_H
