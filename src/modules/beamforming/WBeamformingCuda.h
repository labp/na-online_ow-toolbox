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

#ifndef WBEAMFORMINGCUDA_H_
#define WBEAMFORMINGCUDA_H_

#include <string>
#include <boost/shared_ptr.hpp>
#include <cublas.h>
#include <cuda.h>
#include "core/data/WLDataTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "WBeamforming.h"

//  __global__ void squareElements(float *a, float *b, int N) {
//        /* which element does this compute? */
//        int tid = blockDim.x * blockIdx.x + threadIdx.x;
//
//        /* if valid, squre the array element */
//        if (tid < N)
//                b[tid] = (a[tid]*a[tid]);
//    }
//    __global__ void RowSum(float* B, float* Sum, int N, int M)
//    {
//        int i = blockDim.x * blockIdx.x + threadIdx.x;
//        int j = blockDim.y * blockIdx.y + threadIdx.y;
//        if (i < N && j < M)
//            C[j] += B[i][j];
//    }*/
class WBeamformingCuda: public WBeamforming
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WBeamformingCuda > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WBeamformingCuda > ConstSPtr;

    static const std::string CLASS;

    WBeamformingCuda();
    virtual ~WBeamformingCuda();

    virtual bool calculateBeamforming( const WLMatrix::MatrixT& leadfield , const Eigen::MatrixXcd& CSD, double reg );

    virtual WLEMDSource::SPtr beam( WLEMData::ConstSPtr emd );

private:
    template< typename T >
    static inline void cublasTgemm( char transa, char transb, int m, int n, int k, T alpha, const T* A, int lda, const T* B,
                    int ldb, T beta, T* C, int ldc );
    template< typename T >
       static inline void cublasTgeam(  cublasOperation_t transa, cublasOperation_t transb,int m, int n,const T* A,
                                int lda,const T* B, int ldb,T* C, int ldc,const T* alpha,const T* beta);
    template< typename T >
           static inline void cublasTinverse(  int n,T* A[],int lda,T* C[],int ldc,int batchSize);

    ScalarT* m_A_dev; // m_beam
    ScalarT* m_B_dev;   //m_data

    bool m_beamChanged;
};

template< >
inline void WBeamformingCuda::cublasTgemm< float >( char transa, char transb, int m, int n, int k, float alpha,
                const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc )
{
    cublasSgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

template< >
inline void WBeamformingCuda::cublasTgemm< double >(  char transa, char transb, int m, int n, int k, double alpha,
                const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc )
{
    cublasDgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

template< >
inline void WBeamformingCuda::cublasTgeam< float >( cublasOperation_t transa, cublasOperation_t transb,int m, int n,const float *A,
                int lda,const float *B, int ldb,float *C, int ldc,const float* alpha,const float* beta)
{

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgeam( handle, transa, transb, m, n, alpha, A, lda, beta,  B, ldb, C, ldc );
    cublasDestroy_v2(handle);
}
template< >
inline void WBeamformingCuda::cublasTgeam< double >(  cublasOperation_t transa, cublasOperation_t transb,int m, int n,const double *A,
                int lda,const double *B, int ldb,double *C, int ldc,const double* alpha,const double* beta )
{


    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasDgeam( handle, transa, transb, m, n, alpha, A, lda, beta,  B, ldb, C, ldc );
    cublasDestroy_v2(handle);
}
template< >
inline void WBeamformingCuda::cublasTinverse< float >(int n,float* A[],int lda,float* C[],int ldc,int batchSize)
{
    int *PivotArray;
    int *infoArray;
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasSgetrfBatched(handle, n, A, lda, PivotArray, infoArray, batchSize);
    cublasSgetriBatched(handle, n, A, lda, PivotArray, C, ldc, infoArray, batchSize);
    cublasDestroy_v2(handle);
}
template< >
inline void WBeamformingCuda::cublasTinverse< double >(int n,double* A[],int lda,double* C[],int ldc,int batchSize)
{
    int PivotArray;
    int infoArray;
    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    cublasDgetrfBatched(handle, n, A, lda, &PivotArray, &infoArray, batchSize);
    cublasDgetriBatched(handle, n, A, lda, &PivotArray, C, ldc, &infoArray, batchSize);
    cublasDestroy_v2(handle);

}



#endif  // WBEAMFORMINGCUDA_H_
