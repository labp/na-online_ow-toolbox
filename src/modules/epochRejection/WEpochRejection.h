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

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"

class WEpochRejection: public boost::enable_shared_from_this< WEpochRejection >
{
public:

    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WEpochRejection > SPtr;

    /**
     * Destructor
     */
    virtual ~WEpochRejection();

    /**
     * This method receives the processing thresholds.
     */
    virtual void setThresholds( double eegLevel, double eogLevel, double megGrad, double megMag );

    /**
     * Proceeds the rejection of the all modalities for the given input based on the
     * user defined level values.
     *
     * \return A boolean value, which specifies, whether or not the input object has to reject.
     */
    virtual bool doRejection( const WLEMMeasurement::ConstSPtr emm ) = 0;

    /**
     * Defines the number of rejections for the current input.
     */
    virtual size_t getCount() const;

protected:

    /**
     * Constructor
     */
    WEpochRejection();

    /**
     * Method to separate valid modalities from invalid modalities.
     *
     * \return false, if the modality has to skip else true.
     */
    virtual bool validModality( LaBP::WEModalityType::Enum modalityType );

    virtual double getThreshold( LaBP::WEModalityType::Enum modalityType, size_t channelNo);

    double m_eegThreshold;

    double m_eogThreshold;

    double m_megGradThreshold;

    double m_megMagThreshold;

    size_t m_rejCount;
};

#endif /* WEPOCHREJECTION_H_ */
