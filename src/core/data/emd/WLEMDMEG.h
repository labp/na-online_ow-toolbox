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

#include <core/common/WAssert.h>
#include <core/common/WDefines.h> // OW_API_DEPRECATED
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/container/WLArrayList.h"
#include "core/data/WLMegCoilInfo.h"
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
    typedef boost::shared_ptr< WLEMDMEG > SPtr; //!< Abbreviation for a shared pointer.

    typedef boost::shared_ptr< const WLEMDMEG > ConstSPtr; //!< Abbreviation for const shared pointer.

    typedef std::vector< size_t > CoilPicksT; //!< Collection of indices for coil picks.

    static const std::string CLASS; //!< Class name for logging purpose.

    WLEMDMEG();

    explicit WLEMDMEG( const WLEMDMEG& meg );

    explicit WLEMDMEG( WLEModality::Enum modality );

    virtual ~WLEMDMEG();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    /**
     * Returns the positions in millimeter. TODO(pieloth): Which unit, meter or millimeter?
     *
     * \return Positions in millimeter.
     */
    WLArrayList< WPosition >::SPtr getChannelPositions3d();

    /**
     * Returns the positions in millimeter. TODO(pieloth): Which unit, meter or millimeter?
     *
     * \return Positions in millimeter.
     */
    WLArrayList< WPosition >::ConstSPtr getChannelPositions3d() const;

    /**
     * Sets the positions.
     *
     * \note Positions must be in millimeter. TODO(pieloth): Which unit, meter or millimeter?
     * \param chanPos3d Positions to set.
     */
    void setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d );

    /**
     * Gets the coil information.
     *
     * \return Coil information.
     */
    WLArrayList< WLMegCoilInfo::SPtr >::SPtr getCoilInformation();

    /**
     * Gets the coil information for a coil.
     *
     * \param idx Coil index.
     * \throws WOutOfBounds
     * \return Coil information at index idx.
     */
    WLMegCoilInfo::SPtr getCoilInformation( WLArrayList< WLMegCoilInfo::SPtr >::size_type idx );

    /**
     * Gets the coil information for a coil.
     *
     * \param idx Coil index.
     * \throws WOutOfBounds
     * \return Coil information at index idx.
     */
    WLMegCoilInfo::ConstSPtr getCoilInformation( WLArrayList< WLMegCoilInfo::SPtr >::size_type idx ) const;

    /**
     * Sets the coil information.
     *
     * \param coilInfos Coil information to set.
     */
    void setCoilInformation( WLArrayList< WLMegCoilInfo::SPtr >::SPtr coilInfos );

    static bool createCoilInfos( WLEMDMEG* const meg );

    /**
     * Returns the faces.
     *
     * \return List of faces.
     */
    WLArrayList< WVector3i >::SPtr getFaces();

    /**
     * Returns the faces.
     *
     * \return List of faces.
     */
    WLArrayList< WVector3i >::ConstSPtr getFaces() const;

    /**
     * \deprecated Please use extractCoilModality()
     * \param type Coil types.
     * \return List containing requested coils
     */
    OW_API_DEPRECATED
    WLArrayList< WVector3i >::ConstSPtr getFaces( WLEMEGGeneralCoilType::Enum type ) const;

    /**
     * Sets the faces.
     *
     * \param faces Faces to set.
     */
    void setFaces( WLArrayList< WVector3i >::SPtr faces );

    /**
     * Sets the faces.
     *
     * \deprecated Please use setFaces( WLArrayList< WVector3i >::SPtr faces )
     * \param faces Faces to set.
     */
    OW_API_DEPRECATED
    void setFaces( boost::shared_ptr< std::vector< WVector3i > > faces );

    /**
     * Gets the coil coordinate system x-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::SPtr getEx();

    /**
     * Gets the coil coordinate system x-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::ConstSPtr getEx() const;

    /**
     * Sets the coil coordinate system x-axis unit vector.
     *
     * \param vec Unit vector.
     */
    void setEx( WLArrayList< WVector3f >::SPtr vec );

    /**
     * Gets the coil coordinate system y-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::SPtr getEy();

    /**
     * Gets the coil coordinate system y-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::ConstSPtr getEy() const;

    /**
     * Sets the coil coordinate system y-axis unit vector.
     *
     * \param vec Unit vector.
     */
    void setEy( WLArrayList< WVector3f >::SPtr vec );

    /**
     * Gets the coil coordinate system z-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::SPtr getEz();

    /**
     * Gets the coil coordinate system z-axis unit vector.
     *
     * \return A unit vector.
     */
    WLArrayList< WVector3f >::ConstSPtr getEz() const;

    /**
     * Sets the coil coordinate system z-axis unit vector.
     *
     * \param vec Unit vector.
     */
    void setEz( WLArrayList< WVector3f >::SPtr vec );

    /**
     * \deprecated Please use coilPicks().
     * \param channelId
     * \return
     */
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
    WLEModality::Enum m_modality; //!< MEG, MEG_MAG, MEG_GRAD or MEG_GRAD_MERGE.

    WLArrayList< WPosition >::SPtr m_chanPos3d; //!< Channel positions.

    WLArrayList< WLMegCoilInfo::SPtr >::SPtr m_coilInfos; //!< Coil information.

    WLArrayList< WVector3i >::SPtr m_faces; //!< Channel faces/triangulation.

    WLArrayList< WVector3f >::SPtr m_eX; //!< Coil coordinate system x-axis unit vector.
    WLArrayList< WVector3f >::SPtr m_eY; //!< Coil coordinate system y-axis unit vector.
    WLArrayList< WVector3f >::SPtr m_eZ; //!< Coil coordinate system z-axis unit vector.

    /**
     * Mutable to reset the picks after a data change and lazy load.
     * \deprecated
     */
    OW_API_DEPRECATED
    mutable std::vector< size_t > m_picksMag;

    /**
     * Mutable to reset the picks after a data change and lazy load.
     * \deprecated
     */
    OW_API_DEPRECATED
    mutable std::vector< size_t > m_picksGrad;
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
