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
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/util/profiler/WLTimeProfiler.h"
//#include <core/common/exceptions/WPreconditionNotMet.h>
//#include "core/util/profiler/WLProfilerLogger.h"
//#include "core/util/profiler/WLTimeProfiler.h"
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
{
    wlog::debug( CLASS ) << "CPU called!";
    WLTimeProfiler prfTime( CLASS, "beam" );
//m_beam hat Gewichtung Beamformer
    MatrixT A = emd->getData();
    MatrixT E=MatrixT::Zero(m_beam->rows(),A.cols());
    MatrixT D=MatrixT::Zero(m_beam->rows(),A.cols());
    MatrixT C=MatrixT::Zero(m_beam->rows(),A.cols());

    E= *m_beam* A;        //Matrixmultiplikation Gewichtung und Daten
//    E=E.array()*E.array();
//    C=E.rowwise().sum();
//    D=C.replicate(1,E.cols());


//Ausgabe als Source
    m_result.reset( new MatrixT(E) );
    WLEMData::DataSPtr Beam(new WLEMData::DataT (*m_result));
    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
    emdOut->setData(Beam);

    return emdOut;
}
