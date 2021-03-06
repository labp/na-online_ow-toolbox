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

#ifndef WMEMMGENERATOR_H_
#define WMEMMGENERATOR_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * Generates a EEG measurement with random data.
 *
 * \author pieloth
 * \ingroup io
 */
class WMEMMGenerator: public WModule
{
public:
    WMEMMGenerator();
    virtual ~WMEMMGenerator();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void connectors();

    virtual void properties();

    virtual void moduleInit();

    virtual void moduleMain();

private:
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; //!<  Output connector for buffered input connectors.

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropDouble m_propSamplFreq; //!< Sampling frequency in Hz

    WPropInt m_propLength; //!< Length in seconds.

    WPropInt m_propChans;

    WPropTrigger m_trgGenerate;

    void hdlTrgGenerate();

    bool generateEMM();

    WLEMMeasurement::SPtr m_emm;

    struct EDataStatus
    {
        enum Enum
        {
            NO_DATA, DATA_GENERATED, GENERATING_DATA, DATA_ERROR
        };

        static std::string name( Enum val );
    };
    EDataStatus::Enum m_dataStatus;
    WPropString m_propDataStatus;
    void updateDataStatus( EDataStatus::Enum status );
};

inline void WMEMMGenerator::updateDataStatus( EDataStatus::Enum status )
{
    m_dataStatus = status;
    m_propDataStatus->set( EDataStatus::name( status ), true );
}

#endif  // WMEMMGENERATOR_H_
