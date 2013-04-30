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

#ifndef WLEMDMEG_H
#define WLEMDMEG_H

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/emd/WLEMD.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

namespace LaBP
{
    class WLEMDMEG: public WLEMD
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WLEMDMEG > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WLEMDMEG > ConstSPtr;

        WLEMDMEG();

        explicit WLEMDMEG( const WLEMDMEG& meg );

        virtual ~WLEMDMEG();

        virtual WLEMD::SPtr clone() const;

        virtual WEModalityType::Enum getModalityType() const;

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

        /**
         * NOTE: The method does not modify any object data, but Ex may modified indirectly!
         */
        std::vector< WVector3f >& getEx() const;
        void setEx( boost::shared_ptr< std::vector< WVector3f > > vec );

        /**
         * NOTE: The method does not modify any object data, but Ey may modified indirectly!
         */
        std::vector< WVector3f >& getEy() const;
        void setEy( boost::shared_ptr< std::vector< WVector3f > > vec );

        /**
         * NOTE: The method does not modify any object data, but Ez may modified indirectly!
         */
        std::vector< WVector3f >& getEz() const;
        void setEz( boost::shared_ptr< std::vector< WVector3f > > vec );

    private:
        boost::shared_ptr< std::vector< WPosition > > m_chanPos3d;

        boost::shared_ptr< std::vector< WVector3i > > m_faces;

        boost::shared_ptr< std::vector< WVector3f > > m_eX;
        boost::shared_ptr< std::vector< WVector3f > > m_eY;
        boost::shared_ptr< std::vector< WVector3f > > m_eZ;

        /*
         * member contains absolute position of channel with coordinate system in this position
         * TODO(fuchs): Definition der Speicherung der Kanalpositionen und des zugehörig. Koord.-systems
         *
         * HPI
         *
         * number of coils used to track the head position
         * uint8_t m_nrHpiCoils;
         *
         * name of corresponding HPI eventchannel
         * std::string m_eventChanName;
         *
         *
         * vector<Coils>
         * Coils: uint8_t m_nr;
         *        int32_t m_bitmask;
         *        float m_freq;
         */
    };
}

#endif  // WLEMDMEG_H
