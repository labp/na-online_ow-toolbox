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

// NOTE: Needs Eigen v3.1 or higher for sparse matrices, see README
#ifndef WSOURCERECONSTRUCTION_H_
#define WSOURCERECONSTRUCTION_H_

#include <string>
#include <set>

#include <boost/shared_ptr.hpp>
#include <Eigen/SparseCore>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

using Eigen::SparseMatrix;

class WSourceReconstruction
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WSourceReconstruction > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WSourceReconstruction > ConstSPtr;

    typedef WLEMData::SampleT ScalarT;

    typedef WLEMData::DataT MatrixT;

    typedef WLEMData::DataSPtr MatrixSPtr;

    typedef SparseMatrix< ScalarT > SpMatrixT;

    typedef boost::shared_ptr< SpMatrixT > SpMatrixSPtr;

    struct WEWeightingCalculation
    {
        enum Enum
        {
            MN, WMN
        };

        static std::set< Enum > values();

        static std::string name( Enum value );
    };

    WSourceReconstruction();

    virtual ~WSourceReconstruction();

    virtual void reset();

    void setLeadfield( MatrixSPtr matrix );

    const MatrixT& getLeadfield() const;

    bool hasLeadfield() const;

    bool calculateWeightningMatrix( WSourceReconstruction::WEWeightingCalculation::Enum type );

    const SpMatrixT& getWeighting() const;

    bool hasWeighting() const;

    virtual bool calculateInverseSolution( const MatrixT& noiseCov, const MatrixT& dataCov, double snr );

    const MatrixT& getInverse() const;

    bool hasInverse() const;

    virtual WLEMDSource::SPtr reconstruct( WLEMData::ConstSPtr emd ) = 0;

    static WLEMDSource::SPtr createEMDSource( WLEMData::ConstSPtr emd, MatrixT matrix );

    static bool averageReference( WLEMData::DataT& dataOut, const WLEMData::DataT& dataIn );

protected:
    MatrixSPtr m_leadfield;

    SpMatrixSPtr m_weighting;

    MatrixSPtr m_inverse;
};

#endif  // WSOURCERECONSTRUCTION_H_
