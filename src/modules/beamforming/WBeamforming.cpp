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

using std::set;
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



    m_beam.reset();


}

void WBeamforming::setSource( size_t source )
{

    //m_value = source;
}



bool WBeamforming::hasBeam() const
{
    return m_beam.get() != NULL;
}

bool WBeamforming::calculateBeamforming(const WLMatrix::MatrixT&   data, const WLMatrix::MatrixT& Leadfield , const WLMatrix::MatrixT& Noise, const WLMatrix::MatrixT& Data  )
{
    //zur übergreifenden Nutzung in WBeamfomingCPU
    //Übergabe der Daten an m_data

            //Kovarianzmatrix des Rauschen als Einheitsmatrix

    //zur übergreifenden Nutzung in WBeamfomingCPU
    //Übergabe der Leadfield an m_leadfield
            m_leadfield.reset( new MatrixT( Leadfield) );
    //Init
            wlog::debug( CLASS ) << "calculateBeamforming() called!";
    //Leadfield Spalten zur Berechnung
            MatrixT leadfield(Leadfield.rows(),Leadfield.cols()); ///TODO
            MatrixT Cdinv; //Daten kovarianzmatrix invertiert
            MatrixT Cd;//=MatrixT::Identity(data.rows(),data.rows()); //Daten Kovarianzmatrix
            //Whitener
             //RauschCov
                MatrixT Cn;//=MatrixT::Identity(Noise.rows(),Noise.rows());
                Cn=Noise;  //Diagonalmatrix

                MatrixT E=MatrixT::Identity(data.rows(),data.rows());
               //Covariance
                Cd=Data;

          //whitening daten
          MatrixT Cm(data.rows(),data.rows());
          MatrixT Cm1(data.rows(),data.rows());
          MatrixT Cm2(data.rows(),data.rows());
          MatrixT Cm3(data.rows(),data.rows());
          MatrixT Proj=MatrixT::Identity(data.rows(),data.rows());
/*          Cm1= Cd*Proj.transpose();
          Cm2=Proj*Cm1;*/
         Cm3=Cd*Cn.transpose();

          Cm=Cn*Cm3;

          //Absolutwerte Daten
         /* MatrixT Dat;
          Dat= data.cwiseAbs();*/
          m_data.reset( new MatrixT( data) );

          //Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm)) //Datenkovarianz
          //Regularisierung
          MatrixT Cdr(data.rows(),data.rows());
                          Cdr=((0.01*Cm.trace())/Cm.rows())*E;
                                    MatrixT CD;
                                    CD= Cm+Cdr;


                  //Inverse DataCov
                              MatrixT Cdinv1;

                               // Cdinv=CD.inverse();//funktioniert nu ohne inverse
                                 Cdinv1= CD.transpose()*CD;
                                  Cdinv= Cdinv1.inverse()*CD.transpose();


          //whitening leadfield
          leadfield=Cn*Leadfield;            //TODO
         // leadfield.array()=Leadfield.array();

            // Leafield transpose matrix
                    MatrixT LT = leadfield.transpose();






    MatrixT LCd(leadfield.cols(),1);
    MatrixT LCdinv(leadfield.cols(),1);
                        MatrixT W(leadfield.cols(),leadfield.rows());

                              MatrixT Help1(leadfield.cols(),1);
            for( int j=0;j<leadfield.cols();j++)

            {
            //Zwischenmatrix=LF_tansponiert*Inverse Daten-Matrix*LF


               // LCd(j,0)= LT.row(j) * Cdinv *leadfield.col(j) ;
                LCd(j,0)= LT.row(j) * Cdinv *leadfield.col(j) ;
                MatrixT LCdT(LCd.rows(),1);
                LCdT.block(j,0,1,1)= LCd.block(j,0,1,1).transpose()*LCd.block(j,0,1,1);

                LCdinv.block(j,0,1,1)= LCdT.block(j,0,1,1)*LCd.block(j,0,1,1).transpose();



                //Gewichtungsmatrix -> ein Wert für jeden Sensor
                //W= Inverse der Zwischenmatrix*LF_transponiert*Daten-Kovarianzmatrix invertiert

                W.row(j)= LCdinv(j,0)*LT.row(j) * Cdinv;



            }




            /*W= W.cwiseAbs();*/
           //            //Normierung
                             MatrixT W5;
                             W5= W.rowwise().norm();
                             MatrixT W2;
                             W2= W5.replicate(1,W.cols());
                             MatrixT W3;
                             W3= W.array()/W2.array();


                                   wlog::debug( CLASS ) << "W " << W.rows() << " x " << W.cols();
            //Ergebnis an m_beam übergeben
                                  //m_beam.reset( new MatrixT( W) );
                                   m_beam.reset( new MatrixT(W3) );
                                   wlog::debug( CLASS ) << "m_beam" << m_beam->rows() << " x " << m_beam->cols();
    return true;
}

