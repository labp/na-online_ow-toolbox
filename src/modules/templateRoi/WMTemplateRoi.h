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

#ifndef WMTEMPLATEROI_H_
#define WMTEMPLATEROI_H_

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <core/graphicsEngine/WROI.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMCommand.h"
#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"
#include "core/util/roi/WLROISelectorSource.h"

/**
 * ROI example module, marks the region in source space.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup misc
 */
class WMTemplateRoi: public WLModuleDrawable
{
public:
    WMTemplateRoi();

    virtual ~WMTemplateRoi();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

protected:
    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

    virtual boost::shared_ptr< WModule > factory() const;

    virtual const char** getXPMIcon() const;

    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr labp );
    virtual bool processReset( WLEMMCommand::SPtr labp );

private:
    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; //!< Input connector.

    boost::shared_ptr< WCondition > m_propCondition; //!< A condition used to notify about changes in several properties.

    WLEMMeasurement::SPtr m_Emm;

    void updateOutput();

    void updateOutput( WLEMMeasurement::SPtr emm );

    void roiChanged();

    void startUp( WLEMMeasurement::SPtr emm );
};

#endif  // WMTEMPLATEROI_H_
