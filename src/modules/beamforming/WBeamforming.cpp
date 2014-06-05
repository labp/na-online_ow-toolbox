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

#include <algorithm> // transform
#include <cmath>
#include <functional> // minus
#include <set>
#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"
//#include "core/util/profiler/WLTimeProfiler.h"

#include "WBeamforming.h"

using Eigen::Triplet;
using std::minus;
using std::set;
using std::transform;
using WLMatrix::MatrixT;
using WLSpMatrix::SpMatrixT;

const std::string WBeamforming::CLASS = "WBeamforming";

WBeamforming::WBeamforming()
{
    // TODO Auto-generated constructor stub

}

WBeamforming::~WBeamforming()
{
    // TODO Auto-generated destructor stub
}

//void WBeamforming::setLeadfieldMEG( WLMatrix::SPtr leadfield )
//{
//    m_leadfield = leadfield;
//    m_beam.reset();
//}

void WBeamforming::reset()
{
//    ExclusiveLockT lock(m_lockData);


    m_beam.reset();


}

void WBeamforming::setSource( size_t source )
{

    m_value = source;
}



bool WBeamforming::hasBeam() const
{
    return m_beam.get() != NULL;
}


bool WBeamforming::calculateBeamforming(const WLMatrix::MatrixT&   data, const WLMatrix::MatrixT& Leadfield  )
{
    //zur übergreifenden Nutzung in WBeamfomingCPU
    //Übergabe der Daten an m_data
            m_data.reset( new MatrixT( data) );

    //zur übergreifenden Nutzung in WBeamfomingCPU
    //Übergabe der Leadfield an m_leadfield
            m_leadfield.reset( new MatrixT( Leadfield) );
    //Init
            wlog::debug( CLASS ) << "calculateBeamforming() called!";
    //Leadfield Spalten zur Berechnung
            MatrixT leadfield(Leadfield.rows(),m_value);
            for(unsigned int j=0;j<m_value;j++)
            {

                leadfield.col(j)=Leadfield.block(0,j,m_leadfield->rows(),1).array();

            }
            wlog::debug( CLASS ) << "l " << leadfield.rows() << " x " << leadfield.cols();

                                                                                                                                                                //leadfield.col(0) = Leadfield.col(m_value)  ;            //eingegebene Wert ist die Spalte der Leadfield=Quelle

    // Leafield transpose matrix
            MatrixT LT = leadfield.transpose();
//   //NOise Matrix
//        MatrixT N;
//
//        const int dRows = data.rows();
//        const int dCols = data.cols();
//
//        WLMatrix::SPtr U1( new MatrixT( dRows, dCols ) );
//        U1->setRandom();
//        WLMatrix::SPtr U2( new MatrixT( dRows, dCols ) );
//        U2->setRandom();
//        U1->array().abs();
//        U2->array().abs();
//        MatrixT U1a = 2 * M_PI * *U1; // Zufallswerte 1 cos(2*PI*u1)
//        MatrixT U2a = U2->array() + 0.1;
//        U2a = U2a.array().log(); //Zufallswere 2 sqrt(-2*ln (u2)); keine 0! wegen ln
//        U2a = U2a.array() - 1.0;
//        N = U1a.array().cos() * U2a.array().sqrt();
//    //Kovarianzmatrix Rauschen
//        MatrixT Cn;
//        MatrixT NT = N.transpose();
//        Cn = N * NT;
            //Kovarianzmatrix des Rauschen als Einheitsmatrix
                    MatrixT Cn=MatrixT::Identity(leadfield.rows(), leadfield.rows());

    //Dipolmoment als Zufallsmenge
            const int MdRows = leadfield.cols();
            const int MdCols = data.cols();
            WLMatrix::SPtr Md( new MatrixT( MdRows, MdCols ) );
            Md->setRandom();                                         //Zufälliges Dipolmoment für jeden Dipol über die Anzahl Abtastwerte
    // Mittel der Momente über die Anzahl Abtastwerte
            MatrixT Mdm = Md->rowwise().mean();				        //Matrix Md(m,1)
            MatrixT Mdmm;
            Mdmm= Mdm.replicate(1, MdCols );                        //Erweitern der Matrix auf Spalten Dipolmoment-> durch Mittelung Spaltenvektor
   //Covarianzmatrix Dipolmoment

            MatrixT D;			                                    //Differenz
            MatrixT Mda;
            Mda= Md->array();
//
                D=Mda - Mdmm;                                       //Differenz Dipolmoment und Mittel der Momente
                MatrixT DT ;
                DT = D.transpose();
                wlog::debug( CLASS ) << "DT " << DT.rows() << " x " << DT.cols();
                wlog::debug( CLASS ) << "D " << D.rows() << " x " << D.cols();
                MatrixT Cm;
         Cm = D*DT;                                                 //Kovarianzmatrix Dipolmoment

    //Kovarianzmarix abhängig von Daten
            MatrixT Cd;
            Cd = leadfield*Cm *LT  + Cn;                            //Daten-Kovarianzmatrix

   // gewichtung für jeden Sensor für jede Quelle
            MatrixT W;
            MatrixT Cdinv = Cd.inverse();                           //Inverse Daten-Kovarianzmatrix

            //Zwischenmatrix=LF_tansponiert*Inverse Daten-Matrix*LF
                MatrixT LCd = LT * Cdinv *leadfield;
                LCd = LCd.inverse();                                //Inverse der Zwischenmatrix

                //Gewichtungsmatrix -> ein Wert für jeden Sensor
                //W= Inverse der Zwischenmatrix*LF_transponiert*Daten-Kovarianzmatrix invertiert
                        W = LCd * LT * Cdinv;			            //Matrix (1,N)
                        MatrixT Wd;
                        W.transposeInPlace();                       //Matrix (N,1)
                        MatrixT Wrep = W.replicate( 1, MdCols );    //Vervielfältigen auf Spalten der Datenmatrix

                        wlog::debug( CLASS ) << "Wrep " << Wrep.rows() << " x " << Wrep.cols();
                //Ergebnis an m_beam übergeben
                        m_beam.reset( new MatrixT( Wrep) );

//    //gewichtung Signal
//        Wd = Wrep.cwiseProduct( data ); // TODO check result
//        wlog::debug( CLASS ) << "  Wd " <<   Wd.rows() << " x " <<   Wd.cols();
//    //Beamforming Signal
//        MatrixT B = Wd.rowwise().sum();         //Spaltenvektor, transponiert ausgeben?
//        m_beam.reset( new MatrixT( B ) );
//        wlog::debug( CLASS ) << "  B " <<   B.rows() << " x " <<   B.cols();
    return true;
}

