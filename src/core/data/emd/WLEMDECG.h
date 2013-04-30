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

#ifndef WLEMDECG_H
#define WLEMDECG_H

#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMEnumTypes.h"

#include "WLEMD.h"

namespace LaBP
{
    class WLEMDECG: public WLEMD
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WLEMDECG > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WLEMDECG > ConstSPtr;

        /**
         * TODO(kaehler): Comments
         */
        WLEMDECG();

        explicit WLEMDECG( const WLEMDECG& ecg );

        /**
         * TODO(kaehler): Comments
         */
        virtual ~WLEMDECG();

        virtual LaBP::WLEMD::SPtr clone() const;

        virtual WEModalityType::Enum getModalityType() const;

        LaBP::WEPolarityType::Enum getPolarityType() const;

        void setPolarityType( LaBP::WEPolarityType::Enum polarityType );

    private:

        /**
         * TODO(kaehler):
         * one commen reference electrode for all channels
         * member contains position for each channel (2d, 3d) the common reference position and the tesselation
         */
        // Unipolar m_unipoar;
        /**
         * TODO(kaehler):
         * member contains a list of positions of electrode pairs
         */
        // Bipoler m_bipolar;
        /**
         * TODO(kaehler): doxygen \ref
         */
        LaBP::WEPolarityType::Enum m_polarityType;
    };
}
#endif  // WLEMDECG_H
