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

#ifndef WMHEADPOSITIONCORRECTION_H_
#define WMHEADPOSITIONCORRECTION_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

#include "WHeadPositionCorrection.h"

/**
 * Corrects the head position, MEG only (in progress).
 *
 * \author pieloth
 * \ingroup forward
 */
class WMHeadPositionCorrection: public WLModuleDrawable
{
public:
    typedef boost::shared_ptr< WMHeadPositionCorrection > SPtr; //!< Abbreviation for a shared pointer.

    WMHeadPositionCorrection();
    virtual ~WMHeadPositionCorrection();

    virtual WModule::SPtr factory() const; // WModule

    virtual const std::string getName() const; // WPrototyped

    virtual const std::string getDescription() const; // WPrototyped

    virtual const char** getXPMIcon() const; // WModule

protected:
    virtual void connectors(); // WModule

    virtual void properties(); // WModule

    virtual void moduleInit(); // WLModuleDrawable

    virtual void moduleMain(); // WModule

    virtual bool processCompute( WLEMMeasurement::SPtr emm ); // WLEMMCommandProcessor

    virtual bool processInit( WLEMMCommand::SPtr cmdIn ); // WLEMMCommandProcessor

    virtual bool processReset( WLEMMCommand::SPtr cmdIn ); // WLEMMCommandProcessor

private:
    WPropGroup m_propGroup;
    WPropDouble m_propMvThreshold;
    WPropDouble m_propRadius;
    WPropPosition m_propPosition;

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Buffered input connector.

    WCondition::SPtr m_propCondition; //!< A condition used to notify about changes in several properties.

    WHeadPositionCorrection m_correction;
};

#endif  // WMHEADPOSITIONCORRECTION_H_
