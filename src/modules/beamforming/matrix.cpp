
#include <iostream>
#include "/opt/include/eigen321/Eigen/Dense"

#include <math.h>

using namespace Eigen;
using namespace std;
#include <iostream>
using namespace std;
using namespace std;

//typedef Eigen::SparseMatrix<double> SpMat;

int main()
{


    MatrixXd Tri(3,3);


    Tri<< 1,2,5,        //-> knotenpunkte der dreiecke- jede zeile ein dreieck, aus knoten 5,2,5
          1,4,2,
          3,4,6;

    int Max= Tri.maxCoeff() ;       //max. laufindex?  Max zeile?

    //SparseMatrix<double> Neigh(Max,Max);
    MatrixXd Neigh=MatrixXd::Zero(Max,Max);
    int Neigh_cols= Max;    //anzahl spalten Neigh


    VectorXi r(Max*Max);
    MatrixXd v(1,Tri.cols());
    MatrixXd e=MatrixXd::Zero(Max,3);
    MatrixXd F=MatrixXd::Zero(1,e.rows()*e.cols());
   MatrixXd g(1,e.rows()*e.cols());
   MatrixXd d=MatrixXd::Zero(Max,Max);
   d.array() +=1;
   cout << "r " <<d << '\n';
    int y;
    int knoten;
    int erg=0;
    int knoten2;
     for(int i=0;i<Tri.rows();i++)
   {
       v.array()=Tri.row(i).array();        //zeile in vektor speichern

       for(int a=0;a<Tri.cols();a++)

       {
           y =v(0,a);
           cout << "y " << y << '\n';
//           for (int j = 0; j<= Max; ++j)
//       {
           for(knoten=0;knoten<Max;knoten++)
           {
               knoten2=knoten;

               if (y==knoten2)

           {
            int f;
               f=knoten2;
               cout << "knoten " <<knoten2 << '\n';

               e.row(f).array()=Tri.row(a);
               g= Map<MatrixXd>(e.data(),1,e.rows()*e.cols());;

//
//       }
               F.array()+=g.array();
                 for(int s=0;s<=e.rows()*e.cols();s++) //for loop to access the first data
                 {
                     for(int t=s+1;t<=e.rows()*e.cols()-1;t++) //for loop to check the first data with the rest of the data in the array
                     {
                         if(g(0,s)==F(0,t))
                         {
                             F(0,t)=0;
                         }
                     }
                 }
           }}}
       cout << "r " <<e << '\n';
         cout << "r " <<g << '\n';
         cout << "F " <<F << '\n';
   }



//
//
//  cout << "r " << F << '\n';
//
//for (int f=0; f<g.cols();f++)
//{
//    int y= g(0,f);
//    if(y!=0 && y!=knoten)
//    {
//        Neigh(knoten,y)+=1;
//    }
//}

cout << "r " <<Neigh << '\n';


return 0;
}
/*int main()
{
    MatrixXd A(6,2);
    A<<1,2,1,2,1,2,1,2,1,2,1,2;
    int sens=3;
    int m_value=2;
    MatrixXd B(A.rows()/sens,A.cols());
    for(unsigned int j=0;j<m_value;j++)
             {

                 B.row(j)=A.block(sens*j,0,sens,A.cols()).colwise().sum();

             }
    cout << "A " <<A << '\n';
    cout << "B " << B << '\n';
    return 0;

}*/
