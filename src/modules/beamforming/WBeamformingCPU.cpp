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

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WBeamformingCPU.h"
using WLMatrix::MatrixT;

const std::string WBeamformingCPU::CLASS = "WBeamformingCPU";

WBeamformingCPU::WBeamformingCPU()
{

}

WBeamformingCPU::~WBeamformingCPU()
{
}

WLEMDSource::SPtr  WBeamformingCPU::beam( WLEMData::ConstSPtr emd  )
{wlog::debug( CLASS ) << "CPU called!";

    //m_beam hat Gewichtung Beamformer
    MatrixT E=MatrixT::Zero(m_leadfield->cols(),m_data->cols());

  E= *m_beam* *m_data;




            m_result.reset( new MatrixT(E) );                                       //BB
            WLEMData::DataSPtr Beam(new WLEMData::DataT (*m_result));
           const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
           emdOut->setData(Beam);

            return emdOut;
}
