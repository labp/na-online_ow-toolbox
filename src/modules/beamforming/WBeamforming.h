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

#ifndef WBEAMFORMING_H_
#define WBEAMFORMING_H_


#include <string>
#include <set>
#include <boost/shared_ptr.hpp>
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/data/WLDataTypes.h"



class WBeamforming
{
public:
    static const std::string CLASS;

    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WBeamforming > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WBeamforming > ConstSPtr;

   WBeamforming();
 //Beamformer type
    struct WEType
          {
              enum Enum
              {
                  DICS, LCMV
              };

              static std::vector< Enum > values();

              static std::string name( Enum value );
          };


    virtual ~WBeamforming();

 //Beamformer
    virtual bool calculateBeamforming(  const WLMatrix::MatrixT& leadfield, const Eigen::MatrixXcd& CSD, double reg);

    bool hasBeam() const;

    void setType( WBeamforming::WEType::Enum value );

//reset beamformer
    virtual void reset();

    virtual WLEMDSource::SPtr beam( WLEMData::ConstSPtr emd  )=0;


protected:



    WLMatrix::SPtr m_beam;      //Gewichtung
    WLMatrix::SPtr m_leadfield; //Leadfield
    WLMatrix::SPtr m_data;		//Datenmatrix
    WLMatrix::SPtr m_result;    //Egebnis

    size_t m_type;

};

#endif /* WBEAMFORMING_H_ */
