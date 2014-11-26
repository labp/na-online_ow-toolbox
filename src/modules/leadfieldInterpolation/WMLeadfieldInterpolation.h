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

#ifndef WMLEADFIELDINTERPOLATION_H_
#define WMLEADFIELDINTERPOLATION_H_

#include <string>

#include <mne/mne_forwardsolution.h>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleInputDataCollection.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

/**
 * TODO
 *
 * \author pieloth
 * \ingroup forward
 */
class WMLeadfieldInterpolation: public WModule, public WLEMMCommandProcessor
{
public:
    WMLeadfieldInterpolation();
    virtual ~WMLeadfieldInterpolation();

    /**
     * \par Description
     * Gives back the name of this module.
     * \return the module's name.
     */
    virtual const std::string getName() const;

    /**
     * \par Description
     * Gives back a description of this module.
     * \return description to module.
     */
    virtual const std::string getDescription() const;

    /**
     * Due to the prototype design pattern used to build modules, this method returns a new instance of this method. NOTE: it
     * should never be initialized or modified in some other way. A simple new instance is required.
     *
     * \return the prototype used to create every module in OpenWalnut.
     */
    virtual WModule::SPtr factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmdIn );
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn );
    virtual bool processTime( WLEMMCommand::SPtr cmdIn );
    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

    virtual void moduleInit();

    /**
     * \par Description
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

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    WCondition::SPtr m_propCondition;

    MNELIB::MNEForwardSolution::SPtr m_fwdSolution;

    WLMatrix::SPtr m_leadfieldInterpolated;

    WLEMMeasurement::SPtr m_emm;

    WPropTrigger m_start;

    WPropString m_status;

    WPropFilename m_hdLeadfieldFile;

    WPropFilename m_fiffFile;

    WLEMMeasurement::SPtr m_fiffEmm;

    bool readFiff( const std::string& fname );

    bool readHDLeadfield( const std::string& fname );

    bool interpolate();

    static const std::string ERROR;

    static const std::string COMPUTING;

    static const std::string SUCCESS;

    static const std::string NONE;

    static const std::string FIFF_OK_TEXT; // Conflicts with mne/fiff/fiff_constants.h

    static const std::string HD_LEADFIELD_OK_TEXT;

    static const std::string READING;

    static const std::string COMMAND;
};

#endif  // WMLEADFIELDINTERPOLATION_H_
