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

#include "WBeamforming.h"
#include "WBeamformingCuda.h"
#define CublasSafeCall( err )     __cublasSafeCall( err, __FILE__, __LINE__ )

inline void __cublasSafeCall( cublasStatus err, const char *file, const int line )
{
    if( err != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "cublasSafeCall() failed at %s:%i with code: %d\n", file, line, err );
    }
}

using WLMatrix::MatrixT;

const std::string WBeamformingCuda::CLASS = "WBeamformingCuda";

WBeamformingCuda::WBeamformingCuda()
{
    m_beamChanged = false;
    m_A_dev = NULL;
    m_B_dev = NULL;

    cublasInit();
}

WBeamformingCuda::~WBeamformingCuda()
{
    if( m_A_dev != NULL )
    {

        CublasSafeCall( cublasFree( m_A_dev ) );
    }
    if( m_B_dev != NULL )
    {

        CublasSafeCall( cublasFree( m_B_dev ) );
    }

    cublasShutdown();
}

bool WBeamformingCuda::calculateBeamforming(const WLMatrix::MatrixT&   data, const WLMatrix::MatrixT& leadfield, const WLMatrix::MatrixT& Noise, const WLMatrix::MatrixT& Data  )
{
    m_beamChanged = WBeamforming::calculateBeamforming(data, leadfield, Noise, Data  );
    return m_beamChanged;
}
/*bool WBeamformingCuda::calculateBeamforming(const WLMatrix::MatrixT&   data, const WLMatrix::MatrixT& leadfield  )
{
    m_beamChanged = WBeamforming::calculateBeamforming(data, leadfield  );
    return m_beamChanged;
}*/

WLEMDSource::SPtr WBeamformingCuda::beam( WLEMData::ConstSPtr emd )
{
    wlog::debug( CLASS ) << "beam() called!";
    WLTimeProfiler tp( CLASS, "beam" );
    if( !m_beam )
    {
        throw WPreconditionNotMet( "No weight matrix set!" );
    }



    // Prepare CUDA profiling //
    float elapsedTime;
    cudaEvent_t startCalc, stopCalc; // computation time

    cudaEventCreate( &startCalc );
    cudaEventCreate( &stopCalc );


    // Initialize matrix dimensions //
    const size_t ROWS_A = m_beam->rows();
    const size_t COLS_A = m_beam->cols();
    // prepare copy later

    const size_t ROWS_B = m_data->rows();
    const size_t COLS_B = m_data->cols();

    const size_t ROWS_E = m_leadfield->cols();
    const size_t COLS_E = m_data->cols();



    // Prepare pointer for cuBLAS //
    // const ScalarT* const A_host; Is needed only once
    //const ScalarT* const B_host = emdData.data();

    // C_host is used for copy out in correct column order: MatrixSPtr == cuBLAS-Matrix
    WLMatrix::SPtr S( new MatrixT( ROWS_E, COLS_E ) );
    ScalarT* E_host = S->data();

    // Scalar* A_dev; Is needed only once
//    ScalarT* B_dev;

    ScalarT* E_dev;


    // Copy in //
    if( m_beam )
    {
        const ScalarT* const A_host = m_beam->data();
        const ScalarT* const B_host = m_data->data();

        if( A_host == NULL )
        {
            cudaEventDestroy( startCalc );
            cudaEventDestroy( stopCalc );
            throw WLBadAllocException( "Could not allocate memory for beam matrix!(m_beam)" );
        }
        if( B_host == NULL )
            {
                cudaEventDestroy( startCalc );
                cudaEventDestroy( stopCalc );
                throw WLBadAllocException( "Could not allocate memory for beam matrix!(m_data)" );
            }


        if( m_A_dev != NULL )
        {
            CublasSafeCall( cublasFree( m_A_dev ) );
        }
        CublasSafeCall( cublasAlloc( ROWS_A * COLS_A, sizeof(ScalarT), ( void** )&m_A_dev ) );
        CublasSafeCall( cublasSetMatrix( ROWS_A, COLS_A, sizeof(ScalarT), A_host, ROWS_A, m_A_dev, ROWS_A ) );

        if( m_B_dev != NULL )
        {
            CublasSafeCall( cublasFree( m_B_dev ) );
        }
        CublasSafeCall( cublasAlloc( ROWS_B * COLS_B, sizeof(ScalarT), ( void** )&m_B_dev ) );
        CublasSafeCall( cublasSetMatrix( ROWS_B, COLS_B, sizeof(ScalarT), B_host, ROWS_B, m_B_dev, ROWS_B ) );

        m_beamChanged = false;
    }

    CublasSafeCall( cublasAlloc( ROWS_E * COLS_E, sizeof(ScalarT), ( void** )&E_dev ) );


    // Call cuBLAS kernel //
    // C_dev = 1.0 * A_dev * A_dev.transpose + 0.0 * C_dev
    // S = G * d
    cudaEventRecord( startCalc, 0 );
//multiplikation
    cublasTgemm< ScalarT >( 'n','n', ROWS_A, COLS_B, COLS_A, 1.0, m_A_dev, ROWS_A, m_B_dev,ROWS_B, 0.0, E_dev, ROWS_E );
    CublasSafeCall( cublasGetError() );



    cudaEventRecord( stopCalc, 0 );
    cudaEventSynchronize( stopCalc );
    cudaEventElapsedTime( &elapsedTime, startCalc, stopCalc );
    WLTimeProfiler prfMatMul( CLASS, "beam_matMul", false );
    prfMatMul.setMilliseconds( elapsedTime );
    wlprofiler::log() << prfMatMul;



    // Copy out //
    WLTimeProfiler prfCopyOut( CLASS, "beam_copyOut", false );
    prfCopyOut.start();
    CublasSafeCall( cublasGetMatrix( ROWS_E, COLS_E, sizeof(ScalarT), E_dev, ROWS_E, E_host, ROWS_E ) );
    prfCopyOut.stop();
    wlprofiler::log() << prfCopyOut;

    // Clean memory //
    // CublasSafeCall( cublasFree( m_A_dev ) ); Is done in destructor
    CublasSafeCall( cublasFree( E_dev ) );

    cudaEventDestroy( startCalc );
    cudaEventDestroy( stopCalc );

    // free( A_host ); Do not free, because points to shared pointer
    // free( B_host ); Do not free, because points to shared pointer
    // free( C_host ); Do not free, because points to shared pointer

    const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
    emdOut->setData( S );

    return emdOut;
}
