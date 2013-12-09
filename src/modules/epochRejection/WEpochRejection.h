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

#ifndef WEPOCHREJECTION_H_
#define WEPOCHREJECTION_H_

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"

class WEpochRejection
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochRejection > SPtr;

    WEpochRejection();

    virtual ~WEpochRejection();

    void initRejection();

    void setLevels( double eegLevel, double eogLevel, double megGrad, double megMag );

    /**
     * Proceeds the rejection of the all modalities for the given input based on the
     * user defined level values.
     *
     * \return A boolean value, which specifies, whether or not the input object has to reject.
     */
    bool getRejection( const WLEMMeasurement::SPtr emm );

    /**
     * Defines the number of rejections for the current input.
     */
    size_t getCount();

    /**
     * Method to separate valid modalities from invalid modalities.
     *
     * \return false, if the modality has to skip else true.
     */
    bool validModality( WLEModality::Enum modalityType );

private:

    double m_eegLevel;

    double m_eogLevel;

    double m_megGrad;

    double m_megMag;

    size_t m_rejCount;
};

#endif /* WEPOCHREJECTION_H_ */
