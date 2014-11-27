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

#ifndef WMPCA_H
#define WMPCA_H

#include <string>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WPCA.h"

/**
 * Principal component analysis (in progress).
 *
 * \author jones
 * \ingroup analysis
 */
class WMPCA: public WLModuleDrawable
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

protected:
    // ---------------------------------
    // Methods for WLEMMCommandProcessor
    // ---------------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

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
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

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
