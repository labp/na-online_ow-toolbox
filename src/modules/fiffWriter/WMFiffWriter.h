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

#ifndef WMLEADFIELDINTERPOLATION_H_
#define WMLEADFIELDINTERPOLATION_H_

#include <string>

#include <core/kernel/WModule.h>

#include "core/module/WLModuleInputDataCollection.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

#include "writer/WWriterFiff.h"

class WMFiffWriter: public WModule, public WLEMMCommandProcessor
{
public:
    WMFiffWriter();
    virtual ~WMFiffWriter();

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

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processMisc( WLEMMCommand::SPtr labp );
    virtual bool processTime( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

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

private:
    /**
     * Output connector for a EMMCommand dataset
     */
    LaBP::WLModuleInputDataCollection< WLEMMCommand >::SPtr m_input;

    /**
     * Output connector for a EMMCommand dataset
     */
    LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

    WCondition::SPtr m_propCondition;

    WPropFilename m_propFile;

    WPropString m_propFileStatus;

    bool handleFileChanged();

    WWriterFiff::SPtr m_fiffWriter;

    static const std::string ERROR;

    static const std::string OPEN;

    static const std::string NONE;
};

#endif  // WMLEADFIELDINTERPOLATION_H_