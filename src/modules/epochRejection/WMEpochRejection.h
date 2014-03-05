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
 * Module for epoch rejection.
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

    /**
     * Gives back the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * Gives back a description of this module.
     * \return description to module.
     */
    virtual const std::string getDescription() const;

protected:
    virtual void moduleInit();

    /**
     * Entry point after loading the module. Runs in separate thread.
     */
    virtual void moduleMain();

    /**
     * Initialize the connectors this module is using.
     */
    virtual void connectors();

    /**
     * Initialize the properties for this module.
     */
    virtual void properties();

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual boost::shared_ptr< WModule > factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

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
     * @param emm The EMM object for delivering to the output connector.
     */
    void updateOutput( WLEMMeasurement::SPtr emm );

    void handleApplyBufferSize();

    void handleApplyThresholds();

    void handleModuleReset();

    /**
     * Switches the rejection algorithm.
     */
    void callbackRejectionTypeChanged();

    /**
     * Method to test all buffered Epochs for rejection.
     */
    void checkBufferedEpochs();

    bool checkBadChannels( WLEMMeasurement::SPtr emm );

    /**
     * Input connector for a EMM data set.
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

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
    boost::shared_ptr< WItemSelection > m_rejectionType;
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

    static const std::string NO_FILE_LOADED;

    static const std::string LOADING_FILE;

    static const std::string FILE_LOADED;

    static const std::string FILE_ERROR;

    static const double EEG_THRESHOLD;

    static const double EOG_THRESHOLD;

    static const double MEG_MAG_THRESHOLD;

    static const double MEG_GRAD_THRESHOLD;

    static const std::string EEG_LABEL;

    static const std::string EOG_LABEL;

    static const std::string MEG_MAG_LABEL;

    static const std::string MEG_GRAD_LABEL;

};

#endif  // WMEPOCHREJECTION_H
