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

#ifndef WSOURCERECONSTRUCTIONCUDA_H_
#define WSOURCERECONSTRUCTIONCUDA_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/WLMatrixTypes.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "WSourceReconstruction.h"

class WSourceReconstructionCuda: public WSourceReconstruction
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WSourceReconstructionCuda > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WSourceReconstructionCuda > ConstSPtr;

    static const std::string CLASS;

    WSourceReconstructionCuda();
    virtual ~WSourceReconstructionCuda();

    virtual bool calculateInverseSolution( const LaBP::MatrixT& noiseCov, const LaBP::MatrixT& dataCov, double snr );

    virtual WLEMDSource::SPtr reconstruct( WLEMData::ConstSPtr emd );

private:
    LaBP::MatrixElementT* m_A_dev;
    bool m_inverseChanged;
};

#endif  // WSOURCERECONSTRUCTIONCUDA_H_
