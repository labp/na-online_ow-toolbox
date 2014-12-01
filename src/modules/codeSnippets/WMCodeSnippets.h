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

#ifndef WMCODESNIPPETS_H_
#define WMCODESNIPPETS_H_

#include <string>
#include <vector>

#include <core/kernel/WModule.h>

#include "core/data/emd/WLEMData.h"
#include "core/module/WLModuleInputDataCollection.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

/**
 * Module for small test and helper code e.g. imports, exports and more.
 *
 * \author pieloth
 * \ingroup misc
 */
class WMCodeSnippets: public WModule, public WLEMMCommandProcessor
{
public:
    WMCodeSnippets();
    virtual ~WMCodeSnippets();

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

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    WCondition::SPtr m_propCondition;

    bool writeEmdPositions( WLEMMeasurement::ConstSPtr emm );
    bool writeEmdPositions( const std::vector< WPosition >& positions, std::string fname );

    void emulateSinusWave();

    WPropBool m_writePos;

    WPropTrigger m_trgGenerate;

    static void generateSinusWave( WLEMData::DataT* const in, float sr, float f, float amp, float offset = 0 );

    void testExtract( WLEMMeasurement::SPtr emm );
};

#endif  // WMCODESNIPPETS_H_
