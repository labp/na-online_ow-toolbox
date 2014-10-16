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

#ifndef WEPOCHREJECTION_H_
#define WEPOCHREJECTION_H_

#include <string>

#include <boost/enable_shared_from_this.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"

#include "WThreshold.h"

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
     * Defines the number of rejections for the current input.
     */
    virtual size_t getCount() const;

    /**
     * Returns the threshold list.
     *
     * \return The threshold list.
     */
    virtual WThreshold::WThreshold_List_SPtr getThresholds();

    /**
     * Sets the threshold list.
     *
     * \param thresholdList The threshold list.
     */
    virtual void setThresholds( WThreshold::WThreshold_List_SPtr thresholdList );

    /**
     * Proceeds the rejection of the all modalities for the given input based on the
     * user defined thresholds.
     *
     * \return Boolean value, which specifies, whether or not the input object has to reject.
     */
    virtual bool doRejection( const WLEMMeasurement::ConstSPtr emm ) = 0;

    /**
     * Method to reset the process parameter.
     */
    virtual void initRejection() = 0;

    /**
     * Method to separate valid modalities from invalid modalities.
     *
     * \return false, if the modality has to skip else true.
     */
    bool validModality( WLEModality::Enum modalityType );

protected:
    /**
     * Constructor
     */
    WEpochRejection();

    /**
     * Method to return the threshold for the current processing step based on the modality and the channel number.
     *
     * \param modalityType The modality.
     * \param channelNo The channel number.
     * \return Returns the threshold.
     */
    virtual double getThreshold( WLEModality::Enum modalityType, size_t channelNo );

    /**
     * Method to return the threshold for the current processing step based on the modality.
     *
     * \param modalityType The modality.
     * \return The threshold.
     */
    virtual double getThreshold( WLEModality::Enum modalityType );

    virtual void showThresholds();

    /**
     * A list containing the thresholds.
     */
    WThreshold::WThreshold_List_SPtr m_thresholdList;

    /**
     * Counts the number of rejections in on processing step.
     */
    size_t m_rejCount;
};

#endif  // WEPOCHREJECTION_H_
