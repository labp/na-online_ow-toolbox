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

#ifndef WMLEADFIELDINTERPOLATION_H_
#define WMLEADFIELDINTERPOLATION_H_

#include <string>

#include <mne/mne_forwardsolution.h>

#include <core/kernel/WModule.h>

#include "core/data/WLEMMBemBoundary.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/module/WLModuleInputDataCollection.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

/**
 * Interpolates a "low resolution" leadfield from a "high resolution" leadfield using nearest neighbor, EEG only.
 *
 * \author pieloth
 * \ingroup forward
 */
class WMLeadfieldInterpolation: public WModule, public WLEMMCommandProcessor
{
public:
    WMLeadfieldInterpolation();
    virtual ~WMLeadfieldInterpolation();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmdIn );
    virtual bool processMisc( WLEMMCommand::SPtr cmdIn );
    virtual bool processTime( WLEMMCommand::SPtr cmdIn );
    virtual bool processReset( WLEMMCommand::SPtr cmdIn );

    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Buffered input connector.
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; //!<  Output connector for buffered input connectors.

    WCondition::SPtr m_propCondition;

    MNELIB::MNEForwardSolution::SPtr m_fwdSolution;

    WLMatrix::SPtr m_leadfieldInterpolated;

    WLEMMeasurement::SPtr m_emm;

    WPropTrigger m_start;

    WPropString m_status;

    WPropFilename m_hdLeadfieldFile;

    WPropFilename m_fiffFile;

    WLEMMeasurement::SPtr m_fiffEmm;

    bool readFiff( const std::string& fname );

    bool readHDLeadfield( const std::string& fname );

    bool interpolate();
};

#endif  // WMLEADFIELDINTERPOLATION_H_
