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

    enum modality_code
    {
        eegReject, eogReject, gradReject, magReject, eNULL
    };

    /**
     * Method to assign the parsed value to the property members. The properties will be updated in the view.
     *
     * @param valueList The threshold list to assign to the GUI property values.
     */
    void assignNewValues( std::map< std::string, double > valueList );

    /**
     * Tests a given string for a string pattern and returns the equivalent enum object for a better handling.
     *
     * @param inString The string to test.
     * @return The enum object.
     */
    modality_code hashit( std::string const& inString );

    void setThresholds( boost::shared_ptr< std::list< WThreshold > > );

    /**
     * Method for updating the modules output connector and some GUI fields.
     *
     * @param emm The EMM object for delivering to the output connector.
     */
    void upadteOutput( WLEMMeasurement::SPtr emm );

    /**
     * Input connector for a EMM data set.
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

    /**
     * Property Group for the thresholds.
     */
    WPropGroup m_propGrpThresholds;

    /**
     * Property Group for the epoch properties.
     */
    WPropGroup m_propGrpEpoch;

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
     * The rejection process class for testing the all channels in one process.
     */
    WEpochRejection::SPtr m_rejectingTotal;

    /**
     * The rejection process class for testing all channels separately.
     */
    WEpochRejectionSingle::SPtr m_rejectingSingle;

    /**
     * The threshold parser class.
     */
    WThresholdParser::SPtr m_parser;

    WThreshold::WThreshold_List_SPtr m_thresholds;

    /**
     * Map to match a modality with its property field.
     */
    ModalityUIFiledMap_SPtr m_modalityLabelMap;

    /**
     * List containing the thresholds for processing.
     */
    //boost::shared_ptr< std::list< WThreshold > > m_thresholds;
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
