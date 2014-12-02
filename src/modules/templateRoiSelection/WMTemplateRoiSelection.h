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

#ifndef WMTEMPLATEROISELECTION_H_
#define WMTEMPLATEROISELECTION_H_

#include <boost/shared_ptr.hpp>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * ROI example module, visualizes the selected region.
 * \see \cite Maschke2014
 *
 * \author maschke
 */
class WMTemplateRoiSelection: public WLModuleDrawable
{
public:
    WMTemplateRoiSelection();
    virtual ~WMTemplateRoiSelection();

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
    /**
     *
     */
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

private:

    /**
     * Input connector
     */
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;
};

#endif /* WMTEMPLATEROISELECTION_H_ */
