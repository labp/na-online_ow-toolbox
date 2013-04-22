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

#ifndef WDATASETEMMPCA_H
#define WDATASETEMMPCA_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/common/math/linearAlgebra/WVectorFixed.h"

#include "WDataSetEMMEMD.h"
#include "WDataSetEMMEnumTypes.h"

namespace LaBP
{
    class WDataSetEMMPCA: public LaBP::WDataSetEMMEMD
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WDataSetEMMPCA > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WDataSetEMMPCA > ConstSPtr;

        typedef Eigen::MatrixXd MatrixT;

        typedef Eigen::VectorXd VectorT;

        WDataSetEMMPCA();

        explicit WDataSetEMMPCA( const WDataSetEMMPCA& pca );

        explicit WDataSetEMMPCA( const WDataSetEMMEMD& emd );

        virtual ~WDataSetEMMPCA();

        virtual LaBP::WDataSetEMMEMD::SPtr clone() const;

        virtual LaBP::WEModalityType::Enum getModalityType() const;

        void setTransformationMatrix( boost::shared_ptr< MatrixT > );
        MatrixT& getTransformationMatrix();
        void setChannelMeans( boost::shared_ptr< VectorT > );
        VectorT& getChannelMeans();
        void setPreprocessedData( WDataSetEMMEMD::SPtr new_preprocessed_data );
        WDataSetEMMEMD::SPtr getPreprocessedData();

    private:
        boost::shared_ptr< MatrixT > m_transformation_matrix;

        boost::shared_ptr< VectorT > m_channel_means;

        WDataSetEMMEMD::SPtr m_preprocessed_data;
    };
}
#endif  // WDATASETEMMPCA_H
