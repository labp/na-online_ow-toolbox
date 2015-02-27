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

#include "core/data/WLPositions.h"
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

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processMisc( WLEMMCommand::SPtr labp );
    virtual bool processTime( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    WCondition::SPtr m_propCondition;

    bool writeEmdPositions( WLEMMeasurement::ConstSPtr emm );
    bool writeEmdPositions( const WLPositions::PositionsT& positions, std::string fname );

    void emulateSinusWave();

    WPropBool m_writePos;

    WPropTrigger m_trgGenerate;

    static void generateSinusWave( WLEMData::DataT* const in, float sr, float f, float amp, float offset = 0 );

    void testExtract( WLEMMeasurement::SPtr emm );
};

#endif  // WMCODESNIPPETS_H_
