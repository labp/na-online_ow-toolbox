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

#ifndef WDATASETEMMEEG_H
#define WDATASETEMMEEG_H

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "WLEMD.h"

namespace LaBP
{
    class WLEMDEEG: public LaBP::WLEMD
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WLEMDEEG > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WLEMDEEG > ConstSPtr;

        WLEMDEEG();

        explicit WLEMDEEG( const WLEMDEEG& eeg );

        virtual ~WLEMDEEG();

        virtual WLEMD::SPtr clone() const;

        LaBP::WEPolarityType::Enum getPolarityType() const;

        virtual LaBP::WEModalityType::Enum getModalityType() const;

        /**
         * Returns the positions in millimeter. NOTE: The method does not modify any object data, but positions may modified indirectly!
         */
        boost::shared_ptr< std::vector< WPosition > > getChannelPositions3d() const;

        /**
         * Sets the positions. Positions must be in millimeter.
         */
        void setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d );

        /**
         * Returns the faces. NOTE: The method does not modify any object data, but faces may modified indirectly!
         */
        std::vector< WVector3i >& getFaces() const;
        void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

    private:

        boost::shared_ptr< std::vector< WPosition > > m_chanPos3d;

        boost::shared_ptr< std::vector< WVector3i > > m_faces;

        /**
         * TODO(kaehler): unipolar --> Point3D  m_pos[5].x
         * float m_chanPos2D[nrChan][2]
         * float m_referencePos[3]
         * bool m_isAverageReference // false als default value
         */

        /**
         * TODO(kaehler): bipolar
         * float m_posPair[nrChans][2*3] // xyz xyz
         */

        /**
         * type of polarity, can be unipolar or bipolar
         */
        LaBP::WEPolarityType::Enum m_polarityType;
    };
}
#endif  // WDATASETEMMEEG_H
