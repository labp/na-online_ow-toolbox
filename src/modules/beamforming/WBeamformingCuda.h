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

/**
 * \author ehrlich
 */
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

#endif  // WBEAMFORMINGCUDA_H_
