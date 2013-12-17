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

#include <ostream>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/WDefines.h> // OW_API_DEPRECATED
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WLEMData.h"

class WLEMDMEG: public WLEMData
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDMEG > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDMEG > ConstSPtr;

    typedef std::vector< size_t > CoilPicksT;

    WLEMDMEG();

    explicit WLEMDMEG( const WLEMDMEG& meg );

    virtual ~WLEMDMEG();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    /**
     * Returns the positions in millimeter.
     */
    WLArrayList< WPosition >::SPtr getChannelPositions3d();

    /**
     * Returns the positions in millimeter.
     */
    WLArrayList< WPosition >::ConstSPtr getChannelPositions3d() const;

    OW_API_DEPRECATED
    WLArrayList< WPosition >::ConstSPtr getChannelPositions3d( LaBP::WEGeneralCoilType::Enum type ) const;

    /**
     * Sets the positions. Positions must be in millimeter.
     */
    void setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d );

    OW_API_DEPRECATED
    void setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d );

    /**
     * Returns the faces.
     */
    WLArrayList< WVector3i >::SPtr getFaces();

    WLArrayList< WVector3i >::ConstSPtr getFaces() const;

    OW_API_DEPRECATED
    WLArrayList< WVector3i >::ConstSPtr getFaces( LaBP::WEGeneralCoilType::Enum type ) const;

    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    OW_API_DEPRECATED
    void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

    WLArrayList< WVector3f >::SPtr getEx();
    WLArrayList< WVector3f >::ConstSPtr getEx() const;
    void setEx( WLArrayList< WVector3f >::SPtr vec );

    WLArrayList< WVector3f >::SPtr getEy();
    WLArrayList< WVector3f >::ConstSPtr getEy() const;
    void setEy( WLArrayList< WVector3f >::SPtr vec );

    WLArrayList< WVector3f >::SPtr getEz();
    WLArrayList< WVector3f >::ConstSPtr getEz() const;
    void setEz( WLArrayList< WVector3f >::SPtr vec );

    OW_API_DEPRECATED
    LaBP::WEGeneralCoilType::Enum getChannelType( size_t channelId ) const;

    /**
     * Returns the channels indices for the requested coil type.
     *
     * @param meg data to pick from
     * @param type Requested coil type
     * @return An array of indices for the requested coil type
     */
    static CoilPicksT coilPicks( const WLEMDMEG& meg, LaBP::WEGeneralCoilType::Enum type );

    static bool extractCoilModality( WLEMDMEG::SPtr& megOut, WLEMDMEG::ConstSPtr megIn, WLEModality::Enum type,
                    bool dataOnly = false );

    /**
     * Returns the channels indices for the requested coil type.
     *
     * @param type Requested coil type
     * @return An array of indices for the requested coil type
     */
    OW_API_DEPRECATED
    std::vector< size_t > getPicks( LaBP::WEGeneralCoilType::Enum type ) const;

    /**
     * Returns the data of the requested coil type.
     * Due to the copy effort, getPicks() is recommended for channels wise processing.
     *
     * @param type Requested coil type
     * @return New data containing all channels of the requested coil type
     */
    OW_API_DEPRECATED
    DataSPtr getData( LaBP::WEGeneralCoilType::Enum type ) const; // This is a copy of channels, so the data is not changed.

    /**
     * Returns the data of the requested coil type without the bad channels.
     * Due to the copy effort, getPicks() is recommended for channels wise processing.
     *
     * @param type Requested coil type.
     * @return New data containing all channels of the requested coil type with out the bad channels.
     */
    DataSPtr getDataBadChannels( LaBP::WEGeneralCoilType::Enum type ) const;

    using WLEMData::getData;

private:
    WLEModality::Enum m_modality;

    WLArrayList< WPosition >::SPtr m_chanPos3d;

    WLArrayList< WVector3i >::SPtr m_faces;

    WLArrayList< WVector3f >::SPtr m_eX;
    WLArrayList< WVector3f >::SPtr m_eY;
    WLArrayList< WVector3f >::SPtr m_eZ;

    OW_API_DEPRECATED
    mutable std::vector< size_t > m_picksMag; // mutable to reset the picks after a data change and lazy load.
    OW_API_DEPRECATED
    mutable std::vector< size_t > m_picksGrad; // mutable to reset the picks after a data change and lazy load.

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

std::ostream& operator<<( std::ostream &strm, const WLEMDMEG& obj );

inline LaBP::WEGeneralCoilType::Enum WLEMDMEG::getChannelType( size_t channelId ) const
{
    WAssert( channelId < m_data->size(), "Index out of bounds!" );
    // Sequence: GGMGGMGGM ... 01 2 34 5
    if( channelId > 1 && ( channelId - 2 ) % 3 == 0 )
    {
        return LaBP::WEGeneralCoilType::MAGNETOMETER;
    }
    else
    {
        return LaBP::WEGeneralCoilType::GRADIOMETER;

    }
}

#endif  // WLEMDMEG_H
