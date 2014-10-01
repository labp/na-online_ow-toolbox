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

/**
 * Digitized point, i.e. EEG sensor, HPI coil and more.
 */
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
    PointT m_point;

    WLEPointType::Enum m_kind;

    WLIdentT m_ident;
};

/**
 * Overload for streamed output.
 */
inline std::ostream& operator<<( std::ostream &strm, const WLDigPoint& obj )
{
    strm << obj.CLASS << ": kind=" << obj.getKind() << "; ident=" << obj.getIdent() << "; point=(" << obj.getPoint() << ")";
    return strm;
}

#endif  // WLDIGPOINT_H_
