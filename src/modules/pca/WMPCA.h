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

#ifndef WMPCA_H
#define WMPCA_H

#include <string>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

#include "WPCA.h"

/**
 * This module implements several onscreen status displays
 * \ingroup modules
 */
class WMPCA: public LaBP::WLModuleDrawable
{
public:
    /**
     * standard constructor
     */
    WMPCA();

    /**
     * destructor
     */
    virtual ~WMPCA();

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
    virtual void initModule();

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
    // GUI event handler
    void callbackPCATypeChanged();
    void callbackProcessModalityChanged();

    boost::shared_ptr< WPCA > m_pca;

    WPropGroup m_propGrpPCAComputation;
    WPropInt m_finalDimensions;
    WPropBool m_reverse;

    /**
     * Input connector for a WEEG2 dataset to get filtered
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * Output connector for a filtered WEEG2 dataset
     */
    LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

    /**
     * A condition used to notify about changes in several properties.
     */
    boost::shared_ptr< WCondition > m_propCondition;

    /**
     * The property to know whether use CUDA or not while runtime
     */
    WPropBool m_propUseCuda;

    boost::shared_ptr< WItemSelection > m_processModality;
    WPropSelection m_processModalitySelection;

    /**
     * Lock to prevent concurrent threads trying to update the data-vector
     */
    boost::shared_mutex m_updateLock;
};

#endif  // WMPCA_H
