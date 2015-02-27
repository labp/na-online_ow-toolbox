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

#ifndef WMFIFFWRITER_H_
#define WMFIFFWRITER_H_

#include <string>

#include <core/kernel/WModule.h>

#include "core/module/WLModuleInputDataCollection.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/module/WLEMMCommandProcessor.h"

#include "writer/WWriterFiff.h"

/**
 * Writes an EEG/MEG data stream into a FIFF file (in progress).
 *
 * \author pieloth
 * \ingroup io
 */
class WMFiffWriter: public WModule, public WLEMMCommandProcessor
{
public:
    WMFiffWriter();
    virtual ~WMFiffWriter();

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
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Buffered input connector.
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; //!<  Output connector for buffered input connectors.

    WCondition::SPtr m_propCondition;

    WPropFilename m_propFile;

    WPropTrigger m_trgClose;

    WPropString m_propFileStatus;

    bool hdlFileChanged();

    void hdlFileClose();

    WWriterFiff::SPtr m_fiffWriter;
};

#endif  // WMFIFFWRITER_H_
