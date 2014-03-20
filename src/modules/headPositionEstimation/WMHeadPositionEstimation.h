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

#ifndef WMHEADPOSITIONESTIMATION_H_
#define WMHEADPOSITIONESTIMATION_H_

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

/**
 * Estimation of the head position using the continuous signals of HPI coils.
 *
 * \ingroup modules
 * \author pieloth
 */
class WMHeadPositionEstimation: public WLModuleDrawable
{
public:
    WMHeadPositionEstimation();
    virtual ~WMHeadPositionEstimation();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void connectors();

    virtual void properties();

    virtual void moduleInit();

    virtual void moduleMain();

    virtual bool processInit( WLEMMCommand::SPtr cmdIn );

    virtual bool processCompute( WLEMMeasurement::SPtr emmIn );

    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

private:
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    WCondition::SPtr m_condition; /**< Used to notify module when a property changed. */

    WPropGroup m_propGroup; /**< Collects all properties of the module. */

    WPropDouble m_propHpi1Freq; /**< Frequency for HPI coil 1 in Hz. */
    WPropDouble m_propHpi2Freq; /**< Frequency for HPI coil 2 in Hz. */
    WPropDouble m_propHpi3Freq; /**< Frequency for HPI coil 3 in Hz. */
    WPropDouble m_propHpi4Freq; /**< Frequency for HPI coil 4 in Hz. */
    WPropDouble m_propHpi5Freq; /**< Frequency for HPI coil 5 in Hz. */

    WPropDouble m_propWindowsSize; /**< Windows size in milliseconds. */
    WPropDouble m_propStepSize; /**< Step size in milliseconds. */

    WPropString m_propStatus; /**< Status of the module. */

    WPropTrigger m_trgApplySettings; /**< Forces an apply of the HPI frequencies. */

    /**
     * Applies the frequencies of the HPI coils.
     *
     * \return true, if successful.
     */
    bool handleApplyFreq();

    static const double HPI1_FREQ; /**< Default frequency (sfreq < 600Hz) for HPI coil 1 in Hz. */
    static const double HPI2_FREQ; /**< Default frequency (sfreq < 600Hz) for HPI coil 2 in Hz. */
    static const double HPI3_FREQ; /**< Default frequency (sfreq < 600Hz) for HPI coil 3 in Hz. */
    static const double HPI4_FREQ; /**< Default frequency (sfreq < 600Hz) for HPI coil 4 in Hz. */
    static const double HPI5_FREQ; /**< Default frequency (sfreq < 600Hz) for HPI coil 5 in Hz. */
    static const double WINDOWS_SIZE; /**< Default windows size in millisecnds. */
    static const double STEP_SIZE; /**< Default step size in millisecnds. */

    static const std::string STATUS_OK; /**< Indicates the module status is ok. */
    static const std::string STATUS_ERROR; /**< Indicates an error in module. */
};

#endif  // WMHEADPOSITIONESTIMATION_H_
