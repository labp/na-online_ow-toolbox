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

#ifndef WLBOUNDCALCULATOR_H_
#define WLBOUNDCALCULATOR_H_

#include "core/data/WLMatrixTypes.h"
#include "core/data/WLDataSetEMM.h"
#include "core/data/WLEMMEnumTypes.h"
#include "core/data/emd/WLEMD.h"

namespace LaBP
{
    class WLBoundCalculator
    {
    public:
        explicit WLBoundCalculator( LaBP::WLEMD::SampleT alpha = 1.5 );
        LaBP::WLEMD::SampleT getMax2D( LaBP::WLDataSetEMM::ConstSPtr emm, LaBP::WEModalityType::Enum modality );
        LaBP::WLEMD::SampleT getMax3D( LaBP::WLDataSetEMM::ConstSPtr emm, LaBP::WEModalityType::Enum modality );
        LaBP::WLEMD::SampleT getMax( const MatrixT& matrix );
        LaBP::WLEMD::SampleT getMax( const LaBP::WLEMD::DataT& data );
        virtual ~WLBoundCalculator();

    private:
        LaBP::WLEMD::SampleT m_alpha;
    };

} /* namespace LaBP */
#endif  // WLBOUNDCALCULATOR_H_
