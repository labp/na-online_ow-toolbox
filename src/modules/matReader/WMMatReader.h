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

#ifndef WMMATREADER_H_
#define WMMATREADER_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include "core/kernel/WModule.h"

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * Reads a MATLAB MAT-file into EMM structure.
 *
 * \author pieloth
 */
class WMMatReader: public WModule
{
public:
    WMMatReader();
    virtual ~WMMatReader();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void connectors();

    virtual void properties();

    virtual void moduleInit();

    virtual void moduleMain();

    virtual bool processCompute( WLEMMeasurement::SPtr emm );

private:
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropString m_status;

    WPropFilename m_propMatFile;

    bool handleMatFileChanged();

    WLMatrix::SPtr m_matrix;

    WPropFilename m_propSensorFile;

    bool handleSensorFileChanged();

    boost::shared_ptr< std::vector< WPosition > > m_sensorPos;

    WPropTrigger m_trgGenerate;

    WPropDouble m_propSamplFreq;

    bool handleGenerateEMM();
};

#endif  // WMMATREADER_H_
