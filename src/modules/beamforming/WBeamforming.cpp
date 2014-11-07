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

#include <cmath>
#include <set>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WBeamforming.h"

using WLMatrix::MatrixT;

const std::string WBeamforming::CLASS = "WBeamforming";

WBeamforming::WBeamforming()
{
    m_type = 0;
}

WBeamforming::~WBeamforming()
{
}

void WBeamforming::reset()
{
    m_beam.reset();
}

void WBeamforming::setType( WBeamforming::WEType::Enum value )
{
    m_type = value;
}

std::set< WBeamforming::WEType::Enum > WBeamforming::WEType::values()
{
    std::set< WBeamforming::WEType::Enum > values;
    values.insert( WEType::DICS );
    values.insert( WEType::LCMV );
    return values;
}

std::string WBeamforming::WEType::name( WBeamforming::WEType::Enum value )
{
    switch( value )
    {
        case WEType::DICS:
            return "DICS Beamformer";
        case WEType::LCMV:
            return "LCMV Beamformer";
        default:
            WAssert( false, "Unknown type!" );
            return "ERROR: Undefined!";
    }
}

bool WBeamforming::calculateBeamforming( const WLMatrix::MatrixT& Leadfield, const Eigen::MatrixXcd& CSD, double reg )
{
    wlog::debug( CLASS ) << __func__ << "() called!";

    WLTimeProfiler prfTime( CLASS, __func__ );

//    m_leadfield.reset( new MatrixT( Leadfield ) );
//    m_data.reset( new MatrixT( data ) );

    wlog::debug( CLASS ) << "reg=" << reg;
    switch( m_type )
    {
        case 1:
        {
            wlog::debug( CLASS ) << "LCMV called!";

            MatrixT Data;
            Data = CSD.real();

            // Init

            // Matrizen
            MatrixT leadfield( Leadfield.rows(), Leadfield.cols() );
            MatrixT Cdinv;
            MatrixT E = MatrixT::Identity( Data.rows(), Data.cols() );
            MatrixT W( leadfield.cols(), leadfield.rows() );
            MatrixT Cdr( leadfield.rows(), leadfield.rows() );
            MatrixT CD;

            MatrixT Cd;
            Cd = Data;
            Cdr = ( ( reg * Cd.trace() ) / Cd.rows() ) * E;
            CD = Cd + Cdr;

            leadfield = Leadfield;

            // Pseudoinverse Datenkovarainz
            MatrixT Cdinv1;

            Cdinv1 = CD.transpose() * CD;
            Cdinv = Cdinv1.inverse() * CD.transpose();

            // Leadfield transponiert
            MatrixT LT;
            LT = leadfield.transpose();

            // Beamformer
            for( int j = 0; j < leadfield.cols(); j++ )
            {
                // Zwischenmatrix
                double LCd;
                LCd = LT.row( j ) * Cdinv * leadfield.col( j );

                MatrixT LCdT( 1, Cdinv.rows() );
                LCdT = LT.row( j ) * Cdinv;

                // Gewichtungsmatrix
                MatrixT LCD;

                W.row( j ) = LCdT.array() / LCd;
            }
            // Normierung
            MatrixT WNorm;
            MatrixT WRep;
            MatrixT WBeam;
            WNorm = W.rowwise().norm();
            WRep = WNorm.replicate( 1, W.cols() );
            WBeam = W.array() / WRep.array();

            // Ergebnis an m_beam übergeben
            m_beam.reset( new MatrixT( WBeam ) );
            wlog::debug( CLASS ) << "m_beam LCMV" << m_beam->rows() << " x " << m_beam->cols();
            return true;
        }
        case 0:
        {
            wlog::debug( CLASS ) << "DICS called!";
            Eigen::MatrixXcd Data;
            Data = CSD;

            Eigen::MatrixXcd leadfield( Leadfield.rows(), Leadfield.cols() );
            Eigen::MatrixXcd Cdinv;
            Eigen::MatrixXcd E = Eigen::MatrixXcd::Identity( Data.rows(), Data.cols() );
            Eigen::MatrixXcd W( leadfield.cols(), leadfield.rows() );
            Eigen::MatrixXcd Cdr( leadfield.rows(), leadfield.rows() );
            Eigen::MatrixXcd CD;
            Eigen::MatrixXcd Cd;
            Cd = Data;

            Cdr = ( ( reg * Cd.real().trace() ) / Cd.rows() ) * E;
            CD = Cd + Cdr;                       //TODO regularization

            leadfield.real() = Leadfield;

            // Pseudoinverse Datenkovarainz
            Eigen::MatrixXcd Cdinv1;

            Cdinv1 = CD.transpose() * CD;
            Cdinv = Cdinv1.inverse() * CD.transpose();

            // Leadfield transponiert
            Eigen::MatrixXcd LT;
            LT = leadfield.transpose();

            // Beamformer
            for( int j = 0; j < leadfield.cols(); j++ )
            {
                // Zwischenmatrix
                Eigen::MatrixXcd LCd;
                LCd = LT.row( j ) * Cdinv * leadfield.col( j );

                Eigen::MatrixXcd LCdT( 1, Cdinv.rows() );
                LCdT = LT.row( j ) * Cdinv;
                // Gewichtungsmatrix
                Eigen::MatrixXcd LCD;
                Eigen::MatrixXcd LL;
                LL = LCd.replicate( 1, LCdT.cols() );
                W.row( j ) = LCdT.array() / LL.array();
            }

            // Normierung
            MatrixT WNorm;
            MatrixT WRep;
            MatrixT WBeam;
            MatrixT Bea;
            Bea = W.real();

            WNorm = Bea.rowwise().norm();
            WRep = WNorm.replicate( 1, Bea.cols() );
            WBeam = Bea.array() / WRep.array();

            // Ergebnis an m_beam übergeben
            m_beam.reset( new MatrixT( WBeam ) );
            wlog::debug( CLASS ) << "m_beam" << m_beam->rows() << " x " << m_beam->cols();

            return true;
        }
    }
}

bool WBeamforming::hasBeam() const
{
    return ( m_beam );
}
