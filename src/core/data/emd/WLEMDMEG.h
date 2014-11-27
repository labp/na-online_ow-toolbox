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
#include "core/data/enum/WLEMEGGeneralCoilType.h"

#include "WLEMData.h"

/**
 * MEG data and related measurement information.
 *
 * \author kaehler
 * \ingroup data
 */
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

    explicit WLEMDMEG( WLEModality::Enum modality );

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
    WLArrayList< WPosition >::ConstSPtr getChannelPositions3d( WLEMEGGeneralCoilType::Enum type ) const;

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
    WLArrayList< WVector3i >::ConstSPtr getFaces( WLEMEGGeneralCoilType::Enum type ) const;

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
    WLEMEGGeneralCoilType::Enum getChannelType( WLChanIdxT channelId ) const;

    /**
     * Returns the channels indices for the requested coil type.
     *
     * \param meg data to pick from
     * \param type Requested coil type
     * \return An array of indices for the requested coil type
     */
    static CoilPicksT coilPicks( const WLEMDMEG& meg, WLEMEGGeneralCoilType::Enum type );

    static bool extractCoilModality( WLEMDMEG::SPtr& megOut, WLEMDMEG::ConstSPtr megIn, WLEModality::Enum type, bool dataOnly =
                    false );

    /**
     * Returns the channels indices for the requested coil type.
     *
     * \param type Requested coil type
     * \return An array of indices for the requested coil type
     */
    OW_API_DEPRECATED
    std::vector< size_t > getPicks( WLEMEGGeneralCoilType::Enum type ) const;

    /**
     * Returns the data of the requested coil type.
     * Due to the copy effort, getPicks() is recommended for channels wise processing.
     *
     * \param type Requested coil type
     * \return New data containing all channels of the requested coil type
     */
    OW_API_DEPRECATED
    DataSPtr getData( WLEMEGGeneralCoilType::Enum type ) const; // This is a copy of channels, so the data is not changed.

    /**
     * Returns the data of the requested coil type without the bad channels.
     * Due to the copy effort, getPicks() is recommended for channels wise processing.
     *
     * \param type Requested coil type.
     * \return New data containing all channels of the requested coil type with out the bad channels.
     */
    DataSPtr getDataBadChannels( WLEMEGGeneralCoilType::Enum type ) const;

    /**
     * Returns the data of the requested coil type without the bad channels.
     * Due to the copy effort, getPicks() is recommended for channels wise processing.
     *
     * \param type Requested coil type.
     * \return New data containing all channels of the requested coil type with out the bad channels.
     */
    DataSPtr getDataBadChannels( WLEMEGGeneralCoilType::Enum type, ChannelListSPtr badChans ) const;

    /**
     * Returns the number of bad channels for the given coil type.
     *
     * \param type The coil type.
     * \return The number of bad channels.
     */
    size_t getNrBadChans( WLEMEGGeneralCoilType::Enum type ) const;

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
};

inline WLEMEGGeneralCoilType::Enum WLEMDMEG::getChannelType( WLChanIdxT channelId ) const
{
    WAssert( channelId < m_data->size(), "Index out of bounds!" );
    // Sequence: GGMGGMGGM ... 01 2 34 5
    if( channelId > 1 && ( channelId - 2 ) % 3 == 0 )
    {
        return WLEMEGGeneralCoilType::MAGNETOMETER;
    }
    else
    {
        return WLEMEGGeneralCoilType::GRADIOMETER;
    }
}

inline std::ostream& operator<<( std::ostream &strm, const WLEMDMEG& obj )
{
    const WLEMData& emd = static_cast< const WLEMData& >( obj );
    strm << emd;
    strm << ", positions=" << obj.getChannelPositions3d()->size();
    strm << ", faces=" << obj.getFaces()->size();
    strm << ", ex=" << obj.getEx()->size();
    strm << ", ey=" << obj.getEy()->size();
    strm << ", ez=" << obj.getEz()->size();
    return strm;
}

#endif  // WLEMDMEG_H
