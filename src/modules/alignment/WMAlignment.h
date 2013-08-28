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

#include <Eigen/Core>

#include <core/common/WCondition.h>
#include <core/common/WPropertyTypes.h>

#include "core/module/WLModuleDrawable.h"
#include "core/module/WLModuleInputDataRingBuffer.h"

class WMAlignment: public WLModuleDrawable
{
public:
    WMAlignment();
    virtual ~WMAlignment();

    virtual const std::string getName() const;

    virtual const std::string getDescription() const;

    virtual WModule::SPtr factory() const;

    virtual const char** getXPMIcon() const;

protected:
    virtual void moduleInit();

    virtual void moduleMain();

    virtual void connectors();

    virtual void properties();

    // ----------------------------
    // Methods from WLEMMCommandProcessor
    // ----------------------------
    virtual bool processCompute( WLEMMeasurement::SPtr emm );
    virtual bool processInit( WLEMMCommand::SPtr cmd );
    virtual bool processReset( WLEMMCommand::SPtr cmd );

private:
    typedef Eigen::Matrix< float, 4, 4 > PCLMatrixT;

    /**
     * Input connector for a EMM dataset
     */
    LaBP::WLModuleInputDataRingBuffer< WLEMMCommand >::SPtr m_input;

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

    // ----------------------------
    // Alignment methods
    // ----------------------------
    struct Fiducial
    {
        WPosition lpa;
        WPosition nasion;
        WPosition rpa;
    };

    bool extractFiducialPoints( Fiducial*const eegPoints, const WLEMMeasurement& emm );

    bool estimateTransformation( PCLMatrixT* const trans, const Fiducial& eegPoints, const Fiducial& skinPoints,
                    const WLEMMeasurement& emm );

    bool icpAlign( PCLMatrixT* const trans, double* const score, const WLEMMeasurement& emm, int maxIterations );
};

#endif  // WMALIGNMENT_H_
