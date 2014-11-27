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

#ifndef WMMATWRITER_H_
#define WMMATWRITER_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/io/WLWriterMAT.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

/**
 * Writes EMD object into a MATLAB MAT-file.
 *
 * \author pieloth
 * \ingroup io
 */
class WMMatWriter: public WModule
{
public:
    WMMatWriter();
    virtual ~WMMatWriter();

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
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */

    /**
     * A condition used to notify about changes in several properties.
     */
    WCondition::SPtr m_propCondition;

    WPropString m_status;

    WPropSelection m_selModality;

    WPropFilename m_propMatFile;

    bool handleMatFileChanged();

    WLWriterMAT::SPtr m_writer;

    bool writeData( WLEMMeasurement::ConstSPtr emmIn );
};

#endif  // WMMATWRITER_H_
