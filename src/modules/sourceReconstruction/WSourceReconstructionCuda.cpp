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

#include <cublas.h>
#include <cuda_runtime.h>   // time measurement

#include <cstddef>  // NULL macro
#include <cstdio>   // stderr stream
#include <cstdlib>  // malloc
#include <string>   // CLASS variable

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLMatrixTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WSourceReconstructionCuda.h"

#define CublasSafeCall( err )     __cublasSafeCall( err, __FILE__, __LINE__ )

inline void __cublasSafeCall( cublasStatus err, const char *file, const int line )
{
    if( err != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "cublasSafeCall() failed at %s:%i with code: %d\n", file, line, err );
    }
}

const std::string WSourceReconstructionCuda::CLASS = "WSoureReconstructionCuda";

WSourceReconstructionCuda::WSourceReconstructionCuda()
{
    m_inverseChanged = false;
    m_A_dev = NULL;
}

WSourceReconstructionCuda::~WSourceReconstructionCuda()
{
    if( m_A_dev != NULL )
    {
        CublasSafeCall( cublasFree( m_A_dev ) );
    }
}

bool WSourceReconstructionCuda::calculateInverseSolution( const LaBP::MatrixT& noiseCov, const LaBP::MatrixT& dataCov,
                double snr )
{
    m_inverseChanged = WSourceReconstruction::calculateInverseSolution( noiseCov, dataCov, snr );
    return m_inverseChanged;
}

WLEMDSource::SPtr WSourceReconstructionCuda::reconstruct( WLEMData::ConstSPtr emd )
{
    wlog::debug( CLASS ) << "reconstruct() called!";
    if( !m_inverse )
    {
        // TODO(pieloth): return code
        wlog::error( CLASS ) << "No inverse matrix set!";
    }

    WLTimeProfiler tp( CLASS, "reconstruct" );

    float elapsedTime;
    cudaEvent_t startCalc, stopCalc; // computation time
    cudaEventCreate( &startCalc );
    cudaEventCreate( &stopCalc );

    const size_t ROWS_A = m_inverse->rows();
    const size_t COLS_A = m_inverse->cols();
    // prepare copy later

    const size_t ROWS_B = emd->getNrChans();
    const size_t COLS_B = emd->getData().front().size();

    // ElementT* inverse = ( ElementT* )malloc( 1 * sizeof(ElementT) );
    LaBP::MatrixElementT* B_host = ( LaBP::MatrixElementT* )malloc( ROWS_B * COLS_B * sizeof( LaBP::MatrixElementT ) );
    if( B_host == NULL )
    {
        // TODO(pieloth): return code
        wlog::error( CLASS ) << "Could not allocate memory for EMD data!";
    }

    WLEMData::DataT emdData;
    WSourceReconstruction::averageReference( emdData, emd->getData() );

    // convert from row-major order to column-major order
    WLTimeProfiler prfToMatrix( CLASS, "reconstruct_toMat", false );
    prfToMatrix.start();
    size_t i = 0;
    for( size_t col = 0; col != COLS_B; ++col )
    {
        for( size_t row = 0; row < ROWS_B; ++row )
        {
            // convert from row-major order to column-major order
            B_host[i++] = emdData[row][col];
        }
    }
    prfToMatrix.stop();
    wlprofiler::log() << prfToMatrix;

    const size_t ROWS_C = ROWS_A;
    const size_t COLS_C = COLS_B;
    // Avoid transfer ElementT* to MatrixT
    // LaBP::MatrixT S( ROWS_C, COLS_C );
    boost::shared_ptr< LaBP::MatrixT > S( new LaBP::MatrixT( ROWS_C, COLS_C ) );
    LaBP::MatrixElementT* C_host = S->data();

    cublasInit();

    // ElementT* A_dev;
    LaBP::MatrixElementT* B_dev;
    LaBP::MatrixElementT* C_dev;

    if( m_inverse && m_inverseChanged )
    {
        const LaBP::MatrixElementT* A_host = m_inverse->data();
        if( A_host == NULL )
        {
            // TODO(pieloth): return code
            wlog::error( CLASS ) << "Could not allocate memory for inverse matrix!";
        }
        if( m_A_dev != NULL )
        {
            CublasSafeCall( cublasFree( m_A_dev ) );
        }
        CublasSafeCall( cublasAlloc( ROWS_A * COLS_A, sizeof( LaBP::MatrixElementT ), ( void** )&m_A_dev ) );
        CublasSafeCall( cublasSetMatrix( ROWS_A, COLS_A, sizeof( LaBP::MatrixElementT ), A_host, ROWS_A, m_A_dev, ROWS_A ) );
        m_inverseChanged = false;
    }

    CublasSafeCall( cublasAlloc( ROWS_B * COLS_B, sizeof( LaBP::MatrixElementT ), ( void** )&B_dev ) );
    CublasSafeCall( cublasSetMatrix( ROWS_B, COLS_B, sizeof( LaBP::MatrixElementT ), B_host, ROWS_B, B_dev, ROWS_B ) );

    CublasSafeCall( cublasAlloc( ROWS_C * COLS_C, sizeof( LaBP::MatrixElementT ), ( void** )&C_dev ) );

    // Call cuBLAS kernel
    // C_dev = 1 * A_dev * B_dev + 0 * C_dev
    // S = G * d
    cudaEventRecord( startCalc, 0 );

    cublasSgemm( 'n', 'n', ROWS_A, COLS_B, COLS_A, 1, m_A_dev, ROWS_A, B_dev, ROWS_B, 0, C_dev, ROWS_C );
//    cublasDgemm( CUBLAS_OP_N, CUBLAS_OP_N, ROWS_A, COLS_B, COLS_A, 1, A_dev, ROWS_A, B_dev, ROWS_B, 0, C_dev, ROWS_C );
    CublasSafeCall( cublasGetError() );

    cudaEventRecord( stopCalc, 0 );
    cudaEventSynchronize( stopCalc );
    cudaEventElapsedTime( &elapsedTime, startCalc, stopCalc );
    WLTimeProfiler prfMatMul( CLASS, "reconstruct_matMul", false );
    prfMatMul.setMilliseconds( elapsedTime );
    wlprofiler::log() << prfMatMul;

    WLTimeProfiler prfCopyOut( CLASS, "reconstruct_copyOut", false );
    prfCopyOut.start();
    CublasSafeCall( cublasGetMatrix( ROWS_C, COLS_C, sizeof( LaBP::MatrixElementT ), C_dev, ROWS_C, C_host, ROWS_C ) );
    prfCopyOut.stop();
    wlprofiler::log() << prfCopyOut;

    CublasSafeCall( cublasFree( B_dev ) );
    CublasSafeCall( cublasFree( C_dev ) );

    CublasSafeCall( cublasShutdown() );

    cudaEventDestroy( startCalc );
    cudaEventDestroy( stopCalc );

    // free( A_host ); Do not free, because point to class variable
    free( B_host );
    // free( C_host ); TODO(pieloth): Do not free, because point of return value or copy out??? (depends on eigen impl. check!)

    // const LaBP::WDataSetEMMSource::SPtr emdOut = WSourceReconstruction::createEMDSource( emd, S );
    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) ); // = WSourceReconstruction::createEMDSource( emd, S );
    emdOut->setMatrix( S );

    return emdOut;
}
