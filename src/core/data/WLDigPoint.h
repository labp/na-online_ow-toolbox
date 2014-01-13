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

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>

#include "core/data/WLDataTypes.h"
#include "core/data/enum/WLEPointType.h"
#include "core/data/enum/WLECardinalPoint.h"
#include "core/dataFormat/fiff/WLFiffLib.h"

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

    typedef WPosition PointT;

    static const std::string CLASS;

    WLDigPoint();

    WLDigPoint( const PointT& pos, WLEPointType::Enum kind, WLIdentT ident );

    WLDigPoint( const PointT& pos, WLFiffLib::kind_t kind, WLFiffLib::ident_t ident );

    virtual ~WLDigPoint();

    WLEPointType::Enum getKind() const;

    void setKind( WLEPointType::Enum kind );

    WLIdentT getIdent() const;

    void setIdent( WLIdentT ident );

    const PointT& getPoint() const;

    void setPoint( const PointT& pos );

    /**
     * Checks cardinal point for brain.
     *
     * \return true, if kind is cardinal and ident matches.
     */
    bool checkCardinal( WLECardinalPoint::Enum ident ) const;

private:
    WLEPointType::Enum m_kind;

    WLIdentT m_ident;

    PointT m_point;
};

/**
 * Overload for streamed output.
 */
std::ostream& operator<<( std::ostream &strm, const WLDigPoint& obj );

#endif  // WLDIGPOINT_H_
