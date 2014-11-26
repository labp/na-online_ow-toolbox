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

#ifndef WMALIGNMENT_H_
#define WMALIGNMENT_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/ui/WUIViewWidget.h>
#include <core/kernel/WModule.h>

#include "core/data/WLDataTypes.h"
#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/gui/drawable/WLEMDDrawable3DEEGBEM.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

/**
 * TODO
 *
 * \author pieloth
 * \ingroup forward
 */
class WMAlignment: public WModule, public WLEMMCommandProcessor
{
public:
    WMAlignment();
    virtual ~WMAlignment();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmd );
    virtual bool processReset( WLEMMCommand::SPtr cmd );
    virtual bool processMisc( WLEMMCommand::SPtr cmd );
    virtual bool processTime( WLEMMCommand::SPtr cmd );

private:
    void viewInit();
    void viewUpdate( WLEMMeasurement::SPtr emm );
    void viewReset();

    void moduleInit();

    void handleTrgReset();

    WUIViewWidget::SPtr m_widget;
    WLEMDDrawable3DEEGBEM::SPtr m_drawable;

    WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input; /**< Buffered input connector. */
    WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output; /**<  Output connector for buffered input connectors. */

    WCondition::SPtr m_propCondition;
    WPropTrigger m_trgReset;

    WLMatrix4::Matrix4T m_transformation;

    // ----------------------------
    // Transformation Estimation Properties
    // ----------------------------
    WPropGroup m_propEstGroup;
    WPropPosition m_propEstLPA;
    WPropPosition m_propEstNasion;
    WPropPosition m_propEstRPA;

    // ----------------------------
    // ICP Properties
    // ----------------------------
    WPropGroup m_propIcpGroup;
    WPropInt m_propIcpIterations;
    WPropDouble m_propIcpScore;
    WPropBool m_propIcpConverged;
};

#endif  // WMALIGNMENT_H_
