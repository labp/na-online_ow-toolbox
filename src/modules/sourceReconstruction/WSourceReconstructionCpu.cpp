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

#include "core/data/WLMatrixTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "core/util/WLTimeProfiler.h"

#include "WSourceReconstructionCpu.h"

const std::string WSourceReconstructionCpu::CLASS = "WSourceReconstructionCpu";

WSourceReconstructionCpu::WSourceReconstructionCpu()
{
}

WSourceReconstructionCpu::~WSourceReconstructionCpu()
{
}

WLEMDSource::SPtr WSourceReconstructionCpu::reconstruct( WLEMData::ConstSPtr emd,
                LaBP::WLTimeProfiler::SPtr profiler )
{
    if( !m_inverse )
    {
        // TODO(pieloth): return code
        wlog::error( CLASS ) << "No inverse matrix set!";
    }
    LaBP::WLTimeProfiler::SPtr allProfiler( new LaBP::WLTimeProfiler( CLASS, "reconstruct_all" ) );
    allProfiler->start();

    LaBP::WLTimeProfiler::SPtr avgProfiler( new LaBP::WLTimeProfiler( CLASS, "reconstruct_avgRef" ) );
    avgProfiler->start();
    WLEMData::DataT emdData;
    WSourceReconstruction::averageReference( emdData, emd->getData() );
    avgProfiler->stopAndLog();

    LaBP::WLTimeProfiler::SPtr toMatProfiler( new LaBP::WLTimeProfiler( CLASS, "reconstruct_toMat" ) );
    toMatProfiler->start();
    size_t rows = emdData.size();
    size_t cols = emdData.front().size();
    LaBP::MatrixT data( rows, cols );
    for( size_t r = 0; r < rows; ++r )
    {
        for( size_t c = 0; c < cols; ++c )
        {
            data( r, c ) = emdData[r][c];
        }
    }
    toMatProfiler->stopAndLog();

    LaBP::WLTimeProfiler::SPtr matMulProfiler( new LaBP::WLTimeProfiler( CLASS, "reconstruct_matMul" ) );
    matMulProfiler->start();
    // LaBP::MatrixT S = *m_inverse * data;
    boost::shared_ptr< LaBP::MatrixT > S( new LaBP::MatrixT( *m_inverse * data ) );
    matMulProfiler->stopAndLog();

    // const LaBP::WDataSetEMMSource::SPtr emdOut = WSourceReconstruction::createEMDSource( emd, S );
    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
    emdOut->setMatrix( S );

    if( profiler )
    {
        allProfiler->stopAndLog();
        profiler->addChild( allProfiler );
        profiler->addChild( avgProfiler );
        profiler->addChild( toMatProfiler );
        profiler->addChild( matMulProfiler );
    }

    return emdOut;
}
