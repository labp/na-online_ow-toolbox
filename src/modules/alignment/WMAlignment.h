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

#ifndef WMALIGNMENT_H_
#define WMALIGNMENT_H_

#include <string>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>
#include <core/gui/WCustomWidget.h>
#include <core/gui/WGUI.h>
#include <core/kernel/WModule.h>

#include "core/data/WLEMMCommand.h"
#include "core/data/WLEMMeasurement.h"
#include "core/gui/drawable/WLEMDDrawable3DEEGBEM.h"
#include "core/module/WLEMMCommandProcessor.h"
#include "core/module/WLModuleInputDataRingBuffer.h"
#include "core/module/WLModuleOutputDataCollectionable.h"

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
    virtual bool processMisc( WLEMMCommand::SPtr labp );
    virtual bool processTime( WLEMMCommand::SPtr labp );

private:
    void viewInit();
    void viewUpdate( WLEMMeasurement::SPtr emm );

    void moduleInit();

    WCustomWidget::SPtr m_widget;
    WLEMDDrawable3DEEGBEM::SPtr m_drawable;

    /**
     * Input connector for an EMM Command dataset
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

    /**
     * Output connector for an EMM Command dataset
     */
    LaBP::WLModuleOutputDataCollectionable< WLEMMCommand >::SPtr m_output;

    WCondition::SPtr m_propCondition;
    WPropTrigger m_trgReset;

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
