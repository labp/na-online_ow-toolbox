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

#ifndef WLEMMSUBJECT_H
#define WLEMMSUBJECT_H

#include <map>
#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>
// TODO(pieloth): Deactivated - no setter for birthday
//#include <boost/date_time.hpp>
//#include <boost/date_time/posix_time/posix_time.hpp>
#include <Eigen/Core>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/container/WLList.h"
#include "core/data/enum/WLEModality.h"

#include "WLDataTypes.h"
#include "WLEMMSurface.h"
#include "WLEMMBemBoundary.h"

class WLEMMSubject
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMMSubject > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMMSubject > ConstSPtr;

    WLEMMSubject();

    virtual ~WLEMMSubject();

    std::string getName();
    void setName( std::string name );

    std::string getComment();
    void setComment( std::string comment );

    std::string getHisId();
    void setHisId( std::string hisId );

    WLArrayList< WVector3f >::SPtr getIsotrak();
    WLArrayList< WVector3f >::ConstSPtr getIsotrak() const;
    void setIsotrak( WLArrayList< WVector3f >::SPtr isotrak );

    bool hasSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const;
    WLEMMSurface::SPtr getSurface( WLEMMSurface::Hemisphere::Enum hemisphere );
    WLEMMSurface::ConstSPtr getSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const;
    void setSurface( WLEMMSurface::SPtr surface );

    WLList< WLEMMBemBoundary::SPtr >::SPtr getBemBoundaries();
    WLList< WLEMMBemBoundary::SPtr >::ConstSPtr getBemBoundaries() const;
    void setBemBoundaries( WLList< WLEMMBemBoundary::SPtr >::SPtr bemBoundaries );

    bool hasLeadfield( WLEModality::Enum modality ) const;
    WLMatrix::SPtr getLeadfield( WLEModality::Enum modality );
    WLMatrix::ConstSPtr getLeadfield( WLEModality::Enum modality ) const;
    void setLeadfield( WLEModality::Enum modality, WLMatrix::SPtr leadfield );

private:
    WLArrayList< WVector3f >::SPtr m_isotrak;

    std::string m_name; /**< name of the subject */

    std::string m_comment; /**< comment about subject */

    std::string m_hisId; /**< ID used in the Hospital Information System */

    std::map< WLEMMSurface::Hemisphere::Enum, WLEMMSurface::SPtr > m_surfaces;

    std::map< WLEModality::Enum, WLMatrix::SPtr > m_leadfields;

    WLList< WLEMMBemBoundary::SPtr >::SPtr m_bemBoundaries;
};

inline bool WLEMMSubject::hasSurface( WLEMMSurface::Hemisphere::Enum hemisphere ) const
{
    return m_surfaces.find( hemisphere ) != m_surfaces.end();
}

inline bool WLEMMSubject::hasLeadfield( WLEModality::Enum modality ) const
{
    return m_leadfields.find( modality ) != m_leadfields.end();
}

std::ostream& operator<<( std::ostream &strm, const WLEMMSubject& obj );

#endif  // WLEMMSUBJECT_H
