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

#ifndef WLEMDSOURCE_H_
#define WLEMDSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"

class WLEMDSource: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDSource > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDSource > ConstSPtr;

    static const std::string CLASS;

    WLEMDSource();

    explicit WLEMDSource( const WLEMDSource& source );

    explicit WLEMDSource( const WLEMData& emd );

    virtual ~WLEMDSource();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    WLEModality::Enum getOriginModalityType() const;

    void setOriginModalityType( WLEModality::Enum modality );

private:
    WLEModality::Enum m_originModalityType;
};

#endif  // WLEMDSOURCE_H_
