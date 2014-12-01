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

#ifndef WMDATAAPPENDER_H_
#define WMDATAAPPENDER_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>
#include <core/kernel/WModuleInputData.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMSurface.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * Appends an EMM block by additional data, e.g. source space.
 *
 * \author pieloth
 * \ingroup io
 */
class WMFiffDataAppender: public WModule
{
public:
    WMFiffDataAppender();
    virtual ~WMFiffDataAppender();

    virtual WModule::SPtr factory() const;

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void connectors();

    virtual void properties();

    void moduleInit();

    virtual void moduleMain();

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**< Output connector. */

    WCondition::SPtr m_propCondition; /**< A condition used to notify about changes in several properties. */

    WPropFilename m_srcSpaceFile;
    bool hdlLeadfieldFileChanged( WLMatrix::SPtr* const lf, std::string fName );
    WLEMMSurface::SPtr m_surface;
    bool hdlSurfaceFileChanged( std::string fName );

    WPropFilename m_bemFile;
    WLList< WLEMMBemBoundary::SPtr >::SPtr m_bems;
    bool hdlBemFileChanged( std::string fName );

    WPropFilename m_lfEEGFile;
    WPropFilename m_lfMEGFile;
    WLMatrix::SPtr m_leadfieldEEG;
    WLMatrix::SPtr m_leadfieldMEG;

    WPropTrigger m_trgReset;
    void cbReset();

    WPropString m_propStatus;

    void process( WLEMMCommand::SPtr cmd );
};

#endif  // WMDATAAPPENDER_H_
