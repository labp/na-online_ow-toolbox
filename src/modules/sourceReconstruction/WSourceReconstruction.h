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

#ifndef WSOURCERECONSTRUCTION_H_
#define WSOURCERECONSTRUCTION_H_

#include <string>
#include <set>

#include "boost/thread/mutex.hpp"
#include <boost/thread/locks.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/data/WLDataTypes.h"

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

    void setLeadfield( WLMatrix::SPtr matrix );

    const WLMatrix::MatrixT& getLeadfield() const;

    bool hasLeadfield() const;

    bool calculateWeightningMatrix( WSourceReconstruction::WEWeightingCalculation::Enum type );

    const WLSpMatrix::SpMatrixT& getWeighting() const;

    bool hasWeighting() const;

    virtual bool calculateInverseSolution( const WLMatrix::MatrixT& noiseCov, const WLMatrix::MatrixT& dataCov, double snr );

    const WLMatrix::MatrixT& getInverse() const;

    bool hasInverse() const;

    /**
     * Reconstructs a source distribution.
     *
     * \param emd Inputs data for source reconstruction.
     * \return estimated source distribution
     * \throws WPreconditionNotMet No inverse solution available.
     * \throws WLBadAllocException
     */
    virtual WLEMDSource::SPtr reconstruct( WLEMData::ConstSPtr emd ) = 0;

    static bool averageReference( WLEMData::DataT& dataOut, const WLEMData::DataT& dataIn );

protected:
    typedef boost::shared_mutex MutexT;
    typedef boost::shared_lock< MutexT > SharedLockT;
    typedef boost::unique_lock< MutexT > ExclusiveLockT;

    mutable MutexT m_lockData;

    WLMatrix::SPtr m_leadfield;

    WLSpMatrix::SPtr m_weighting;

    WLMatrix::SPtr m_inverse;
};

#endif  // WSOURCERECONSTRUCTION_H_
