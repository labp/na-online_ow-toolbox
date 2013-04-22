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

#ifndef WDATASETEMMSUBJECT_H
#define WDATASETEMMSUBJECT_H

#include <map>
#include <string>

#include <boost/shared_ptr.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <Eigen/Core>

#include "core/common/math/WLMatrixTypes.h"
#include "core/common/math/linearAlgebra/WVectorFixed.h"
#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "WDataSetEMMEnumTypes.h"
#include "WDataSetEMMSurface.h"
#include "WDataSetEMMBemBoundary.h"

/**
 *
 */
namespace LaBP
{
    /**
     *
     */
    class WDataSetEMMSubject
    {
    public:
        /**
         * Abbreviation for a shared pointer.
         */
        typedef boost::shared_ptr< WDataSetEMMSubject > SPtr;

        /**
         * Abbreviation for const shared pointer.
         */
        typedef boost::shared_ptr< const WDataSetEMMSubject > ConstSPtr;

        /**
         *
         */
        WDataSetEMMSubject();

        /**
         *
         */
        virtual ~WDataSetEMMSubject();

        std::string getName();
        void setName( std::string name );

        boost::shared_ptr< boost::gregorian::date > getBirthday();

        LaBP::WESex::Enum getSex();
        void setSex( LaBP::WESex::Enum sex );

        LaBP::WEHand::Enum getHand();
        void setHand( LaBP::WEHand::Enum hand );

        float getHeight();
        void setHeight( float height );

        float getWeight();
        void setWeight( float weight );

        std::string getComment();
        void setComment( std::string comment );

        std::string getHisId();
        void setHisId( std::string hisId );

        std::vector< WVector3f >& getIsotrak();
        void setIsotrak( boost::shared_ptr< std::vector< WVector3f > > isotrak );

        WDataSetEMMSurface& getSurface( WDataSetEMMSurface::Hemisphere::Enum hemisphere ) const;
        void setSurface( WDataSetEMMSurface::SPtr surface );

        std::vector< WDataSetEMMBemBoundary::SPtr >& getBemBoundaries() const;
        void setBemBoundaries( boost::shared_ptr< std::vector< WDataSetEMMBemBoundary::SPtr > > bemBoundaries );

        MatrixT& getLeadfield( WEModalityType::Enum modality ) const;
        void setLeadfield( WEModalityType::Enum modality, MatrixSPtr leadfield );

    private:

        boost::shared_ptr< std::vector< WVector3f > > m_isotrak;

        /**
         * name of the subject
         */
        std::string m_name;

        /**
         * date of birth of subject
         */
        boost::shared_ptr< boost::gregorian::date > m_birthday;

        /**
         * sex determines whether subject is male, female, or another sex
         */
        LaBP::WESex::Enum m_sex;

        /**
         * hand determines whether subject is left-, right- or both-handed
         */
        LaBP::WEHand::Enum m_hand;

        /**
         * height of subject in m
         */
        float m_height;

        /**
         * weight of subject in kg
         */
        float m_weight;

        /**
         * comment about subject
         */
        std::string m_comment;

        /**
         * ID used in the Hospital Information System
         */
        std::string m_hisId;

        std::map< WDataSetEMMSurface::Hemisphere::Enum, WDataSetEMMSurface::SPtr > m_surfaces;

        std::map< WEModalityType::Enum, MatrixSPtr > m_leadfields;

        boost::shared_ptr< std::vector< WDataSetEMMBemBoundary::SPtr > > m_bemBoundaries;

        // TODO(fuchs): felder ergänzen
        //      leadfield
        //      volume conductor description
        //      evtl. head digitization
    };
}

#endif  // WDATASETEMMSUBJECT_H
