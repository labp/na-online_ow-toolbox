//bem-file einlesen
//in Liste faces-> informationen zu den "Dreiecken"

//Nachbarschaftsmatrix
////---------------------------------------------------------------------------
////
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
#include "/opt/include/eigen321/Eigen/Dense"
//
//#include <core/common/WAssert.h>
//#include <core/common/WLogger.h>

//#include "core/data/emd/WLEMData.h"
//#include "core/data/emd/WLEMDSource.h"
//#include "core/util/profiler/WLTimeProfiler.h"

//#include <boost/smart_ptr/shared_ptr.hpp>
////#include <core/common/exceptions/WTypeMismatch.h>
//#include <core/common/math/linearAlgebra/WMatrixFixed.h>
//#include <core/common/math/linearAlgebra/WPosition.h>
//#include <core/common/math/linearAlgebra/WVectorFixed.h>
//#include <core/common/WLogger.h>
//#include <core/common/WStringUtils.h>
//#include <cstddef>
//#include <vector>
//
//#include "../container/WLArrayList.h"
//#include "../util/WLGeometry.h"


using Eigen::Triplet;
using std::minus;
using std::set;
using std::transform;
using WLMatrix::MatrixT;
using WLSpMatrix::SpMatrixT;
using namespace LaBP;
using namespace std;
//
//SpMatrixT Neighbors(WLList< WLEMMBemBoundary::SPtr >::SPtr FILE, size_t d)
//{
//    MatrixT Tri(3,3);
//    //Tri = FILE;
//    Tri<<
//          5.4,5.5,5.6,
//          4.6,4.6,4.7,
//          3.7,3.8,3.9;
//
//    int Max= Tri.maxCoeff() ;       //max. laufindex?
////    SpMatrixT Neigh(Max,Max)  ;
////    int Neigh_cols= Max;    //anzahl spalten Neigh
////
////    for(int i=0;i<Max;i++)
////    adjacent_find (v.begin (), v.end ());
//    cout<<"No. of duplicates: "<<Max<<endl;
//
//return  Neigh;
//}








//
//#include <iostream>
//using namespace std;
//int main()
//{
//    int array[5]={1,4,4,2.3,4
//                     }; //Initializing array - float werte, int array -> sucht nur nach int
//    int array1[3];
//    int i,j,dup=0;
////    for(j=-1;j<=5;j++) //for loop to access the first data
////    {
//        for(i=0;i<5;i++) //for loop to check the first data with the rest of the data in the array
//        {
//
//                if(array[i]== 4)//array[i])
//                        {
//
//
//                            dup++;
//                            array1[dup]=i;
//                        }
//
//
//       }
////    };
//
//    cout<<"No. of duplicates: "<<dup<<endl;
//    for(int d=0;d<3;d++)
//    cout<<"No. of duplicates: "<<array1[d]<<endl;
//
//    return 0;
//}


int main()
{
    MatrixT Tri(3,3);
    //Tri = FILE;
    Tri<<
          5.4,5.5,5.6,
          4.6,4.6,4.7,
          3.7,3.8,3.9;

    int Max= Tri.maxCoeff() ;       //max. laufindex?
//    SpMatrixT Neigh(Max,Max)  ;
//    int Neigh_cols= Max;    //anzahl spalten Neigh
//
//    for(int i=0;i<Max;i++)
//    adjacent_find (v.begin (), v.end ());
    cout<<"No. of duplicates: "<<Max<<endl;



return 0;
}

























