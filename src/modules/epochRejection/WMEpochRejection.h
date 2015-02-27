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

#ifndef WMEPOCHREJECTION_H
#define WMEPOCHREJECTION_H

#include <list>
#include <map>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"

#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WEpochRejection.h"
#include "WEpochRejectionSingle.h"
#include "WEpochRejectionTotal.h"
#include "WThresholdParser.h"

/**
 * Module for epoch rejection (in progress).
 *
 * \author maschke
 * \ingroup preproc
 */
class WMEpochRejection: public WLModuleDrawable
{
public:
    typedef std::map< WLEModality::Enum, WPropDouble > ModalityUIFiledMap;

    typedef boost::shared_ptr< ModalityUIFiledMap > ModalityUIFiledMap_SPtr;

    /**
     * standard constructor
     */
    WMEpochRejection();

    /**
     * destructor
     */
    virtual ~WMEpochRejection();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

    void updateBadChannels( WLEMMeasurement::SPtr emm );

private:
    void initThresholds();

    void setThresholds( boost::shared_ptr< std::list< WThreshold > > );

    /**
     * Method for updating the modules output connector and some GUI fields.
     *
     * \param emm The EMM object for delivering to the output connector.
     */
    void updateOutput( WLEMMeasurement::SPtr emm );

    void hdlApplyBufferSize();

    void hdlApplyThresholds();

    void hdlModuleReset();

    /**
     * Switches the rejection algorithm.
     */
    void cbRejectionTypeChanged();

    /**
     * Method to test all buffered Epochs for rejection.
     */
    void checkBufferedEpochs();

    bool checkBadChannels( WLEMMeasurement::SPtr emm );

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */

    WCondition::SPtr m_propCondition;

    /**
     * Property group for the rejection module.
     */
    WPropGroup m_propGrpRejection;

    /**
     * Property group for the epoch buffer.
     */
    WPropGroup m_propGrpEpochBuff;

    /**
     * Property Group for the thresholds.
     */
    WPropGroup m_propGrpThresholds;

    /**
     * Property Group for the epoch properties.
     */
    WPropGroup m_propGrpEpoch;

    /**
     * The size for the epoch buffer;
     */
    WPropInt m_epochBufferSize;

    /**
     * Applies the new epoch buffer size to the structure.
     */
    WPropTrigger m_applyBufferSize;

    /**
     * Threshold for the EEG modality.
     */
    WPropDouble m_eegThreshold;

    /**
     * Threshold for the EOG modality.
     */
    WPropDouble m_eogThreshold;

    /**
     * Threshold for the MEG gradiometer channels.
     */
    WPropDouble m_megGradThreshold;

    /**
     * Threshold for the MEG magnetometer channels.
     */
    WPropDouble m_megMagThreshold;

    /**
     * Applies the thresholds for processing.
     */
    WPropTrigger m_applyThresholds;

    /**
     * Integer Property to count the epochs.
     */
    WPropInt m_epochCount;

    /**
     * Integer Property to count the valid epochs.
     */
    WPropInt m_epochCountValid;

    /**
     * Integer Property to count the invalid epochs.
     */
    WPropInt m_epochCountInValid;

    /**
     * File status, if a file was read.
     */
    WPropString m_rejectFileStatus;

    /**
     * File name, if a file was read.
     */
    WPropFilename m_rejectFile;

    /**
     * Integer property to count the bad channels.
     */
    WPropInt m_badChannelCount;

    /**
     * Integer property to count the bad epochs.
     */
    WPropInt m_badEpochCount;

    /**
     * Trigger to reseting the module.
     */
    WPropTrigger m_resetModule;

    /**
     * Selection box to specify the rejection algorithm.
     */
    WItemSelection::SPtr m_rejectionType;
    WPropSelection m_rejectionTypeSelection;

    /**
     * The threshold parser class.
     */
    WThresholdParser::SPtr m_parser;

    /**
     * The threshold list.
     */
    WThreshold::WThreshold_List_SPtr m_thresholds;

    /**
     * Map to match a modality with its property field.
     */
    ModalityUIFiledMap_SPtr m_modalityLabelMap;

    /**
     * Pointer to the rejection algorithm.
     */
    WEpochRejection::SPtr m_rejection;
};

#endif  // WMEPOCHREJECTION_H
