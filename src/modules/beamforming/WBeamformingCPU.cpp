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

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDSource.h"

#include "core/util/profiler/WLProfilerLogger.h"
#include "core/util/profiler/WLTimeProfiler.h"

#include "WBeamformingCPU.h"
using WLMatrix::MatrixT;
const std::string WBeamformingCPU::CLASS = "WBeamformingCPU";

WBeamformingCPU::WBeamformingCPU()
{

}

WBeamformingCPU::~WBeamformingCPU()
{
}

WLEMDSource::SPtr  WBeamformingCPU::beam( WLEMData::ConstSPtr emd  )
{
    //m_beam hat Gewichtung Beamformer
    //Elementweise mit Datenmatix multipliziert
    MatrixT A ;
    MatrixT Mat;
    MatrixT Mat1;
    if (m_value==1)                             //wenn nur ein Dipol interessant
             A =m_beam->cwiseProduct(*m_data);

    else

       {
        Mat = *m_data;
        Mat1=Mat.replicate( 1, m_value );        //bei mehreren Dipolen muss die Datenmatrix vervielfältigt werden verfielfältigung um cols?

         A =m_beam->cwiseProduct(Mat1);         //TODO replicate datenmatrix zeile/spalten-> ausgabe ---anpassung berechnung B--prüfe m_beam !!!
       }
    wlog::debug( CLASS ) << "A " << Mat1.rows() << " x " << Mat1.cols();
    //Anzahl Zeilen von A = Sensoren*Quelle, Spalten= Abtatswerte
    //Addieren aller Zeilen -> ein Beamformersignal für Quelle


            MatrixT B(m_value,A.cols());        //-> fehler, da abtastwerte der verschiedenen quellen hintereinander!!


            for(unsigned int j=0;j<m_value;j++)
            {

                B.row(j)=A.block(m_leadfield->rows()*j,0,m_leadfield->rows(),A.cols()).colwise().sum();

            }


    //if value ==1
//    MatrixT B(1,A.cols());
//    B= A.colwise().sum();
//    MatrixT BB(m_value,m_data->cols());
//    for(unsigned int j=0;j<m_value;j++)
//            {
//
//                BB.row(j)=B.block(0,m_data->cols()*j,1,m_data->cols());
//
//            }
//
//            wlog::debug( CLASS ) << "BB" << BB.rows() << " x " << BB.cols();


    //Null-Matrix erstellen, Zeilen = Anzahl Dipole, Spalten = Anzahl Abtastwerte
//            MatrixT C=MatrixT::Zero(m_leadfield->cols(),m_data->cols() );

    //Übergabe Beamformersignal der Quelle x, an die Zeile x der Matrix             // oder einfach Matrix erstellen mit Größe Leadfield und alles 0 setzen, Spalen der
//            C.row(m_value).array() +=B.row(0).array();                            // interessanten Dipole an entsprechende Spalten der Kopie übergeben,
                                                                                    //damit wäre value = Spalten Leadfield
    //Übergabe Ergebnis als Datenmatrix
//            m_result.reset( new MatrixT(C) );
            m_result.reset( new MatrixT(B) );                                       //BB
            WLEMData::DataSPtr Beam(new WLEMData::DataT (*m_result));
            const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
           emdOut->setData( Beam);



            return emdOut;

    //Ausgabe Beamformer für eine Quelle
            /*    m_result.reset( new MatrixT(B) );//bis hier würde es gehen
            WLEMData::DataSPtr Beam(new WLEMData::DataT (*m_result));
            const WLEMDSource::SPtr emdOut( new WLEMDSource( *emd ) );
            emdOut->setData( Beam);
            wlog::debug( CLASS ) << "B " << B.rows() << " x " << B.cols();
            return emdOut;*/


}
