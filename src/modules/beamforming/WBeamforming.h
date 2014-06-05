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

#include "boost/thread/mutex.hpp"
#include <boost/thread/locks.hpp>
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

    virtual ~WBeamforming();

    virtual bool calculateBeamforming(const WLMatrix::MatrixT&   data, const WLMatrix::MatrixT& leadfield );
//
//    void setLeadfieldMEG( WLMatrix::SPtr leadfield );
    void setSource( size_t source );

    virtual void reset();

    bool hasBeam() const;

//    void setData(WLMatrix::SPtr data);

    virtual WLEMDSource::SPtr beam( WLEMData::ConstSPtr emd  )=0;      //;


protected:
//    typedef boost::shared_mutex MutexT;
//    typedef boost::shared_lock< MutexT > SharedLockT;
//    typedef boost::unique_lock< MutexT > ExclusiveLockT;
/////////////////////////////////////////////////////////////////////////////////////
//    mutable MutexT m_lockData; ///////



    WLMatrix::SPtr m_beam;      //Gewichtung
    WLMatrix::SPtr m_leadfield; //Leadfield
    WLMatrix::SPtr m_data;		//Datenmatrix
    WLMatrix::SPtr m_result;    //Egebnis
    size_t  m_value;            //Anzahl quellen, Spalten


/////////////////////////////////////////////////////////////////////////////////////////
};

#endif /* WBEAMFORMING_H_ */
