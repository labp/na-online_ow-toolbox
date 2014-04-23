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

#ifndef WMEMMGENERATOR_H_
#define WMEMMGENERATOR_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

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
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropDouble m_propSamplFreq;

    WPropInt m_propLength;

    WPropInt m_propChans;

    WPropTrigger m_trgGenerate;

    void handleTrgGenerate();

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
