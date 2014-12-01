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

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WSourceReconstructionCpu.h"

const std::string WSourceReconstructionCpu::CLASS = "WSourceReconstructionCpu";

WSourceReconstructionCpu::WSourceReconstructionCpu()
{
}

WSourceReconstructionCpu::~WSourceReconstructionCpu()
{
}

WLEMDSource::SPtr WSourceReconstructionCpu::reconstruct( WLEMData::ConstSPtr emd )
{
    WLTimeProfiler tp( CLASS, "reconstruct" );
    if( !m_inverse )
    {
        throw WPreconditionNotMet( "No inverse matrix set!" );
    }

    WLEMData::DataT emdData;
    WSourceReconstruction::averageReference( emdData, emd->getData() );

    WLTimeProfiler prfMatMul( CLASS, "reconstruct_matMul", false );
    SharedLockT lock( m_lockData );
    prfMatMul.start();
    WLEMData::DataSPtr S( new WLEMData::DataT( *m_inverse * emdData ) );
    prfMatMul.stop();
    lock.unlock();
    wlprofiler::log() << prfMatMul;

    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
    emdOut->setData( S );

    return emdOut;
}
