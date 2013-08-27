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

#ifndef WLDIGPOINT_H_
#define WLDIGPOINT_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>

class WLDigPoint
{
public:
    /**
     * Convenience typedef for a boost::shared_ptr< WLDigPoint >
     */
    typedef boost::shared_ptr< WLDigPoint > SPtr;

    /**
     * Convenience typedef for a  boost::shared_ptr< const WLDigPoint >
     */
    typedef boost::shared_ptr< const WLDigPoint > ConstSPtr;

    /**
     * FIFF specification: B9. Point definitions
     */
    struct PointType
    {
        enum Enum
        {
            CARDINAL = 1, HPI = 2, EEG = 3, ECG = 3, EXTRA = 4, HEAD_SURFACE = 5, UNKNOWN = -1
        };
    };

    /**
     * FIFF specification: B10. Cardinal points for brain
     */
    struct CardinalPoints
    {
        enum Enum
        {
            LPA = 1, NASION = 2, RPA = 3
        };
    };

    static const std::string CLASS;

    WLDigPoint();

    WLDigPoint( const WPosition& pos, PointType::Enum kind, int ident );

    WLDigPoint( const WPosition& pos, int kind, int ident );

    virtual ~WLDigPoint();

    PointType::Enum getKind() const;

    void setKind( PointType::Enum kind );

    int getIdent() const;

    void setIdent( int ident );

    const WPosition& getPoint() const;

    void setPoint( const WPosition& pos );

    /**
     * Checks cardinal point for brain.
     *
     * \return true, if kind is cardinal and ident matches.
     */
    bool checkCardinal( CardinalPoints::Enum ident ) const;

private:
    PointType::Enum m_kind;

    int m_ident;

    WPosition m_point;
};

#endif  // WLDIGPOINT_H_
