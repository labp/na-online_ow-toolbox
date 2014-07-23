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

#ifndef WMHEADPOSITIONESTIMATION_H_
#define WMHEADPOSITIONESTIMATION_H_

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>
#include <core/ui/WUIGridWidget.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMDHPI.h"
#include "core/gui/drawable/WLEMDDrawable3DHPI.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WContinuousPositionEstimation.h"

#include "WHPISignalExtraction.h"

/**
 * Estimation of the head position using the continuous signals of HPI coils.
 *
 * \ingroup modules
 * \author pieloth
 */
class WMHeadPositionEstimation: public WModule, public WLEMMCommandProcessor
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

    virtual bool processMisc( WLEMMCommand::SPtr cmd );

    virtual bool processTime( WLEMMCommand::SPtr cmd );

private:
    void viewInit();
    void viewUpdate( WLEMMeasurement::SPtr emm );
    void viewReset();

    WUIGridWidget::SPtr m_widget;
    WUIViewWidget::SPtr m_widgetTop;
    WUIViewWidget::SPtr m_widgetSide;
    WUIViewWidget::SPtr m_widgetFront;
    WLEMDDrawable3DHPI::SPtr m_drawableTop;
    WLEMDDrawable3DHPI::SPtr m_drawableSide;
    WLEMDDrawable3DHPI::SPtr m_drawableFront;

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    WCondition::SPtr m_condition; /**< Used to notify module when a property changed. */

    WPropGroup m_propGroupExtraction; /**< Collects all properties for HPI Signal Extraction. */

    WPropDouble m_propHpi1Freq; /**< Frequency for HPI coil 1 in Hz. */
    WPropDouble m_propHpi2Freq; /**< Frequency for HPI coil 2 in Hz. */
    WPropDouble m_propHpi3Freq; /**< Frequency for HPI coil 3 in Hz. */
    WPropDouble m_propHpi4Freq; /**< Frequency for HPI coil 4 in Hz. */
    WPropDouble m_propHpi5Freq; /**< Frequency for HPI coil 5 in Hz. */

    WPropDouble m_propWindowsSize; /**< Windows size in milliseconds. */

    WPropString m_propStatus; /**< Status of the module. */

    WPropTrigger m_trgApplySettings; /**< Forces an apply of the HPI frequencies. */

    /**
     * Applies the frequencies of the HPI coils.
     *
     * \return true, if successful.
     */
    bool handleApplyFreq();

    WHPISignalExtraction::SPtr m_hpiSignalExtraction;

    WPropGroup m_propGroupEstimation; /**< Collects all properties for Head Position Estimation. */

    WPropInt m_propMaxIterations; /**< Maximum iterations for minimization algorithm. */

    WPropDouble m_propEpsilon; /**< Epsilon/threshold for minimization algorithm. */

    WPropDouble m_propInitAlpha; /**< Initial step: alpha angle (degree) for z-y-z rotation. */

    WPropDouble m_propInitBeta; /**< Initial step: beta angle (degree) for z-y-z rotation. */

    WPropDouble m_propInitGamma; /**< Initial step: gamma angle (degree) for z-y-z rotation. */

    WPropDouble m_propInitX; /**< Initial step: x translation (meter). */

    WPropDouble m_propInitY; /**< Initial step: y translation (meter). */

    WPropDouble m_propInitZ; /**< Initial step: z translation (meter). */

    /**
     * Extracts EMDMEG containing magnetometer data only.
     *
     * \param magOut Object to store magnetometer data.
     * \param emmIn EMM containing MEG data.
     * \return True if magnetometer data was extracted.
     */
    bool extractMagnetometer( WLEMDMEG::SPtr& magOut, WLEMMeasurement::ConstSPtr emmIn );

    /**
     *  Extracts the HPI signals from MEG data.
     *
     * \param hpiOut Will store extracted signals.
     * \param magIn Data to extract signals from.
     * \return True if successful, else false.
     */
    bool extractHpiSignals( WLEMDHPI::SPtr& hpiOut, WLEMDMEG::ConstSPtr magIn );

    /**
     * Estimates the head position from the extracted HPI signals.
     *
     * \param magIn Magnetometer data.
     * \param hpiInOut Data to estimate position from.
     * \return True if successful, else false.
     */
    bool estimateHeadPosition( WLEMDHPI::SPtr hpiInOut, WLEMDMEG::ConstSPtr magIn );

    WContinuousPositionEstimation::SPtr m_optim; /**< Algorithm for position estimation. */

    WContinuousPositionEstimation::ParamsT m_lastParams; /**< Transformation parameter of the last estimation, used as initial. */
};

#endif  // WMHEADPOSITIONESTIMATION_H_
