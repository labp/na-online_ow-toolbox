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

#ifndef WMPROFILERLOG_H
#define WMPROFILERLOG_H

#include <string>
#include <fstream>

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

// TODO use OW classes
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "core/dataHandler/WDataSetEMM.h"

/**
 * This module implements several onscreen status displays
 * \ingroup modules
 */
class WMProfilerLog: public WModule
{
public:
    /**
     * standard constructor
     */
    WMProfilerLog();

    /**
     * destructor
     */
    virtual ~WMProfilerLog();

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
    virtual boost::shared_ptr< WModule > factory() const;

    /**
     * Get the icon for this module in XPM format.
     */
    virtual const char** getXPMIcon() const;

private:
    // TODO use OW classes
    /**
     * Input connector for a EMM dataset
     */
    boost::shared_ptr< LaBP::WLModuleInputDataRingBuffer< LaBP::WDataSetEMM > > m_input;

    /**
     * Output connector for a EMM dataset
     */
    boost::shared_ptr< LaBP::WLModuleOutputDataCollectionable< LaBP::WDataSetEMM > > m_output;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

    WPropGroup m_propGrpModule;

    WPropString m_file;

    bool write( std::string fname, LaBP::WDataSetEMM::SPtr emm );

    void write( std::ofstream& fstream, LaBP::WLTimeProfiler::SPtr profiler, std::string prefix );

};

#endif  // WMPROFILERLOG_H
