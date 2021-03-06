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

#include <cstddef>  // NULL macro
#include <cstdio>   // stderr stream
#include <cstdlib>  // malloc
#include <string>   // CLASS variable

#include <cublas.h>
#include <cuda_runtime.h>   // time measurement

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/exception/WLBadAllocException.h"
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

using WLMatrix::MatrixT;

const std::string WSourceReconstructionCuda::CLASS = "WSoureReconstructionCuda";

WSourceReconstructionCuda::WSourceReconstructionCuda()
{
    m_inverseChanged = false;
    m_A_dev = NULL;
    cublasInit();
}

WSourceReconstructionCuda::~WSourceReconstructionCuda()
{
    if( m_A_dev != NULL )
    {
        ExclusiveLockT lock( m_lockData );
        CublasSafeCall( cublasFree( m_A_dev ) );
    }
    cublasShutdown();
}

bool WSourceReconstructionCuda::calculateInverseSolution( const MatrixT& noiseCov, const MatrixT& dataCov, double snr )
{
    m_inverseChanged = WSourceReconstruction::calculateInverseSolution( noiseCov, dataCov, snr );
    return m_inverseChanged;
}

WLEMDSource::SPtr WSourceReconstructionCuda::reconstruct( WLEMData::ConstSPtr emd )
{
    wlog::debug( CLASS ) << "reconstruct() called!";
    WLTimeProfiler tp( CLASS, "reconstruct" );
    if( !m_inverse )
    {
        throw WPreconditionNotMet( "No inverse matrix set!" );
    }

    // Calculare average reference //
    WLEMData::DataT emdData;
    WSourceReconstruction::averageReference( emdData, emd->getData() );

    // Prepare CUDA profiling //
    float elapsedTime;
    cudaEvent_t startCalc, stopCalc; // computation time
    cudaEventCreate( &startCalc );
    cudaEventCreate( &stopCalc );

    SharedLockT lock( m_lockData );
    // Initialize matrix dimensions //
    const size_t ROWS_A = m_inverse->rows();
    const size_t COLS_A = m_inverse->cols();
    // prepare copy later

    const size_t ROWS_B = emd->getNrChans();
    const size_t COLS_B = emd->getSamplesPerChan();

    const size_t ROWS_C = ROWS_A;
    const size_t COLS_C = COLS_B;

    // Prepare pointer for cuBLAS //
    // const ScalarT* const A_host; Is needed only once
    const ScalarT* const B_host = emdData.data();
    // C_host is used for copy out in correct column order: MatrixSPtr == cuBLAS-Matrix
    WLMatrix::SPtr S( new MatrixT( ROWS_C, COLS_C ) );
    ScalarT* C_host = S->data();

    // Scalar* A_dev; Is needed only once
    ScalarT* B_dev;
    ScalarT* C_dev;

    // Copy in //
    if( m_inverse && m_inverseChanged )
    {
        const ScalarT* const A_host = m_inverse->data();
        if( A_host == NULL )
        {
            cudaEventDestroy( startCalc );
            cudaEventDestroy( stopCalc );
            throw WLBadAllocException( "Could not allocate memory for inverse matrix!" );
        }
        if( m_A_dev != NULL )
        {
            CublasSafeCall( cublasFree( m_A_dev ) );
        }
        CublasSafeCall( cublasAlloc( ROWS_A * COLS_A, sizeof( ScalarT ), ( void** )&m_A_dev ) );
        CublasSafeCall( cublasSetMatrix( ROWS_A, COLS_A, sizeof( ScalarT ), A_host, ROWS_A, m_A_dev, ROWS_A ) );
        m_inverseChanged = false;
    }

    CublasSafeCall( cublasAlloc( ROWS_B * COLS_B, sizeof( ScalarT ), ( void** )&B_dev ) );
    CublasSafeCall( cublasSetMatrix( ROWS_B, COLS_B, sizeof( ScalarT ), B_host, ROWS_B, B_dev, ROWS_B ) );

    CublasSafeCall( cublasAlloc( ROWS_C * COLS_C, sizeof( ScalarT ), ( void** )&C_dev ) );

    // Call cuBLAS kernel //
    // C_dev = 1.0 * A_dev * B_dev + 0.0 * C_dev
    // S = G * d
    cudaEventRecord( startCalc, 0 );

    cublasTgemm< ScalarT >( 'n', 'n', ROWS_A, COLS_B, COLS_A, 1.0, m_A_dev, ROWS_A, B_dev, ROWS_B, 0.0, C_dev, ROWS_C );
    CublasSafeCall( cublasGetError() );

    lock.unlock();
    cudaEventRecord( stopCalc, 0 );
    cudaEventSynchronize( stopCalc );
    cudaEventElapsedTime( &elapsedTime, startCalc, stopCalc );
    WLTimeProfiler prfMatMul( CLASS, "reconstruct_matMul", false );
    prfMatMul.setMilliseconds( elapsedTime );
    wlprofiler::log() << prfMatMul;

    // Copy out //
    WLTimeProfiler prfCopyOut( CLASS, "reconstruct_copyOut", false );
    prfCopyOut.start();
    CublasSafeCall( cublasGetMatrix( ROWS_C, COLS_C, sizeof( ScalarT ), C_dev, ROWS_C, C_host, ROWS_C ) );
    prfCopyOut.stop();
    wlprofiler::log() << prfCopyOut;

    // Clean memory //
    // CublasSafeCall( cublasFree( m_A_dev ) ); Is done in destructor
    CublasSafeCall( cublasFree( B_dev ) );
    CublasSafeCall( cublasFree( C_dev ) );

    cudaEventDestroy( startCalc );
    cudaEventDestroy( stopCalc );

    // free( A_host ); Do not free, because points to shared pointer
    // free( B_host ); Do not free, because points to shared pointer
    // free( C_host ); Do not free, because points to shared pointer

    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
    emdOut->setData( S );

    return emdOut;
}
