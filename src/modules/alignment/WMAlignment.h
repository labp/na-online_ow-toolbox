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
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/ui/WCustomWidget.h>
#include <core/kernel/WModule.h>

#include "core/data/WLDataTypes.h"
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
    virtual bool processMisc( WLEMMCommand::SPtr cmd );
    virtual bool processTime( WLEMMCommand::SPtr cmd );

private:
    void viewInit();
    void viewUpdate( WLEMMeasurement::SPtr emm );
    void viewReset();

    void moduleInit();

    void handleTrgReset();

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

    WLMatrix4::Matrix4T m_transformation;

    // ----------------------------
    // Transformation Estimation Properties
    // ----------------------------
    WPropGroup m_propEstGroup;
    WPropPosition m_propEstLPA;
    WPropPosition m_propEstNasion;
    WPropPosition m_propEstRPA;

    static const WPosition LAP;
    static const WPosition NASION;
    static const WPosition RAP;

    // ----------------------------
    // ICP Properties
    // ----------------------------
    WPropGroup m_propIcpGroup;
    WPropInt m_propIcpIterations;
    WPropDouble m_propIcpScore;
    WPropBool m_propIcpConverged;

    static const int ICP_DEFAULT_ITERATIONS;
};

const int WMAlignment::ICP_DEFAULT_ITERATIONS = 10;

// Defaults for intershift is05
//    const WPosition WMAlignment::LAP( -0.0754, -0.0131, -0.0520 );
//    const WPosition WMAlignment::NASION( -0.0012, 0.0836, -0.0526 );
//    const WPosition WMAlignment::RAP( 0.0706, -0.0140, -0.0613 );
// Defaults for hermann
const WPosition WMAlignment::LAP( -0.07286011, 0.018106384, -0.068811984 );
const WPosition WMAlignment::NASION( 0.002131995, 0.098106384, -0.019811981 );
const WPosition WMAlignment::RAP( 0.075132007, 0.017106384, -0.074811978 );

#endif  // WMALIGNMENT_H_
