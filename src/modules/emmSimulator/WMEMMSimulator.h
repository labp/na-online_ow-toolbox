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

#ifndef WMEMMSIMULATOR_H_
#define WMEMMSIMULATOR_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>
#include <core/kernel/WModuleInputData.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleDrawable.h"

/**
 * Simulates a streaming of EMM data. Splits an EMM object into blocks.
 *
 * \author pieloth
 */
class WMEMMSimulator: public WLModuleDrawable
{
public:
    WMEMMSimulator();
    virtual ~WMEMMSimulator();

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

    virtual bool processInit( WLEMMCommand::SPtr cmdIn );

    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

private:
    WModuleInputData< WLEMMCommand >::SPtr m_input; /**< Input connector. */

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropBool m_propAutoStart;
    WPropTrigger m_trgStart;
    WPropTrigger m_trgStop;
    WPropInt m_propBlockSize;
    WPropInt m_propBlocksSent;

    struct EStreaming
    {
        enum Enum
        {
            NO_DATA, READY, STREAMING, STOP_REQUEST
        };
        static std::string name( EStreaming::Enum val );
    };
    EStreaming::Enum m_statusStreaming;
    WPropString m_propStatusStreaming;

    void updateStatus( EStreaming::Enum status );

    void stream();
    void reset();

    WLEMMeasurement::ConstSPtr m_data;

    void hdlTrgStart();
    void cbTrgStop();

    WPropTrigger m_trgReset; /**< Reset additional data. */
    void hdlTrgReset();

    // Additional data
    // ---------------
    WPropGroup m_propGrpAdditional;

    WPropFilename m_srcSpaceFile;
    WLEMMSurface::SPtr m_surface;
    bool hdlSurfaceFileChanged( std::string fName );

    WPropFilename m_bemFile;
    WLList< WLEMMBemBoundary::SPtr >::SPtr m_bems;
    bool hdlBemFileChanged( std::string fName );

    WPropFilename m_lfEEGFile;
    WPropFilename m_lfMEGFile;
    WLMatrix::SPtr m_leadfieldEEG;
    WLMatrix::SPtr m_leadfieldMEG;
    bool hdlLeadfieldFileChanged( std::string fName, WLMatrix::SPtr& lf );

    WPropString m_propStatusAdditional;

    struct EData
    {
        enum Enum
        {
            DATA_NOT_LOADED, DATA_LOADING, DATA_LOADED, DATA_ERROR
        };
        static std::string name( EData::Enum val );
    };

    WLEMMSubject::SPtr m_subject; /**< Stores a copy the input subject. */

    /**
     * If additional data is available, clones the input and sets the data.
     *
     * \param subjectIn Subject to clone
     * \return True if new subject were created and additional data was set.
     */
    bool initAdditionalData( WLEMMSubject::ConstSPtr subjectIn );
};

inline void WMEMMSimulator::updateStatus( EStreaming::Enum status )
{
    m_statusStreaming = status;
    m_propStatusStreaming->set( EStreaming::name( status ), true );
}

#endif  // WMEMMSIMULATOR_H_
