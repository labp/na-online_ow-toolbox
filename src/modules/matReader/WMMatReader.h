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

#ifndef WMMATREADER_H_
#define WMMATREADER_H_

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include "core/kernel/WModule.h"

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMSurface.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

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
    LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropString m_status;

    WPropFilename m_propMatFile;

    bool handleMatFileChanged();

    WLMatrix::SPtr m_matrix;

    WPropTrigger m_trgGenerate;

    WPropDouble m_propSamplFreq;

    bool handleGenerateEMM();

    WPropFilename m_propLfFile;
    WLMatrix::SPtr m_leadfield;
    bool handleLfFileChanged();

    WPropFilename m_propSrcSpaceFile;
    LaBP::WLEMMSurface::SPtr m_surface;
    bool handleSurfaceFileChanged();

    // ***************
    // Status messages
    // ***************
    static const std::string NONE;

    static const std::string ERROR_EMM;
    static const std::string SUCCESS_EMM;
    static const std::string GENERATE_EMM;

    static const std::string ERROR_READ;
    static const std::string SUCCESS_READ;

    static const std::string READING_MAT;
    static const std::string READING_LF;
    static const std::string READING_SRC;

    static const double SAMPLING_FEQUENCY;
};

#endif  // WMMATREADER_H_
