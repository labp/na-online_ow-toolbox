
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <complex>
using namespace Eigen;
using namespace std;

MatrixXd Noise(int N,int T)
{
	MatrixXd No(N,T);
	MatrixXd u1 = MatrixXd::Random(N,T);				//Zufallswerte zwischen -1,1
	MatrixXd u2 = MatrixXd::Random(N,T);

	ArrayXXd u1_a(N,T);
	u1_a= u1.array().abs();
	u1_a=2*M_PI*u1_a;									// Zufallswerte 1 cos(2*PI*u1)
	ArrayXXd u2_a(N,T);
	ArrayXXd N_1(N,T);
	ArrayXXd N_2(N,T);
	u2_a= u2.array().abs();

	u2_a=u2_a+0.1; //Zufallswere 2 sqrt(-2*ln (u2)); keine 0! wegen ln

	u2_a=u2_a.log()-1;		//alle zahlen negativ sonst sqrt(-1)

	u2_a=-2*u2_a;
	N_1=u1_a.cos();
	N_2=u2_a.sqrt();
	//
	No.array()= N_1*N_2;
	return No;
	}
MatrixXd MomDi(MatrixXd Md, int M, int T)
{

	MatrixXd Mdm(M,1);							// Mittel der Momente über die Anzahl Abtastwerte, Vektor [3] x,y,z
	Mdm.array()=Md.rowwise().mean();
	MatrixXd Mdm_(M,T);
	Mdm_=Mdm.replicate(1,T);					//Vervielfäligung Vektor um Anzahl Spalten um T
	return Mdm_;
}

MatrixXd CovMomDi(MatrixXd Md, MatrixXd Mdm, int M, int T)
{
MatrixXd Mc(M,T);
MatrixXd D(M,T);				//Differenzmatrix
D=Md-Mdm;
MatrixXd Dt;	//Transponiert
Dt=D;
Dt.transposeInPlace();
Mc= D*Dt;
return Mc;
}

MatrixXd CovNoi(MatrixXd No)
{
	MatrixXd Noc;
	MatrixXd Not;

	Not=No;
	Not.transposeInPlace();

	Noc=No*Not;
	return Noc;
}

MatrixXd CovD(MatrixXd L,MatrixXd Mc, MatrixXd Noc)
{
	MatrixXd Cy;

	MatrixXd Lt;
	Lt=L;
	Lt.transposeInPlace();
	//
	Cy= L*Mc*Lt+Noc;
	return Cy;
}
MatrixXd Weight(MatrixXd L, MatrixXd Cy)
{
	MatrixXd W;
	MatrixXd Cyi;
	MatrixXd LCi;
	MatrixXd LCm;
	MatrixXd LCmi;
	MatrixXd Lt;
	Lt=L;
	Lt.transposeInPlace();
	Cyi= Cy.inverse();
	LCm= Lt*Cyi*L;
	LCmi= LCm.inverse();
	W= LCmi*Lt*Cyi;
	return W;
}
MatrixXd WeightSig(MatrixXd W,MatrixXd Y, int M, int N, int T)
{
	MatrixXd Wv(1,M*N);
	MatrixXd Wt;
	Wt=W;
	Wt.transposeInPlace();
	Wv= Map<MatrixXd>(Wt.data(),1,N*M);
	cout<<"Wt"<<endl<<Wt<<endl;
	MatrixXd W_r;
	W_r= Wv.replicate(T,1);
	cout<<"W_r"<<endl<<W_r<<endl;
	MatrixXd Yr;
	Yr=Y;
	Yr.transposeInPlace();
	cout<<"Y"<<endl<<Y<<endl;
	cout<<"Yr"<<endl<<Yr<<endl;
	MatrixXd Y_;
	Y_= Yr.replicate(1,M);
	cout<<"Y_"<<endl<<Y_<<endl;

	MatrixXd Y_n;
	Y_n=W_r.array()*Y_.array();			//Gewichtete Signale
	return Y_n;
}





int main ()
{

//Beamforming: Y=LQ+No, Y= Signal der Channels, L=Lead Field Matrix, P = Dipolinformation, No=rauschen
//Benötigte Informationen Einlesen
int N; 												//Anzahl der Channels
int M;   											//Dimension
int T;												//Anzahl der Abtastwerte
//MatrixXd L;										//Lead Field
//MatrixXd No(N,T);  								//Rauschmatrix


//Beispiel zur Überprüfung Code ---- für eine Richtung der Quelle, bsp x, ansonsten immer M*3
N=3;
M= 1;
T=5;
//MatrixXd L= MatrixXd::Random(N,M);
//Lead_field_Marix einzeln einlesen
//-> L_(N,3) 1,2,3 für x,y,z
/////////Daten einlesen
MatrixXd Y= MatrixXd::Random(N,T);
MatrixXd L= MatrixXd::Random(N,M);
//Rauschmatrix einlesen////////////////////////////////////////////////////////////
MatrixXd No(N,T);
No=Noise(N,T);			//cos(2*PI*u1)*sqrt(-2*ln (u2)) /////gaussverteiltes Rauschen - mit Box Müller
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Dipolmoment als Zufallsmenge
MatrixXd Md= MatrixXd::Random(M,T) ;		//Zufälliges Dipolmoment für jeden Dipol über die Anzahl Abtastwerte
MatrixXd Mdm;
Mdm=MomDi(Md,M,T);
/////////////Moment-Kovarianz-Matrix
MatrixXd Mc(M,T);
Mc= CovMomDi(Md, Mdm, M,T);
////////////////////////////////////////////////////////////////////////////////
//Mittel des Rauschens
MatrixXd Nom;
Nom=No.rowwise().mean();
///////////////////////////////////////////////////////////////////////
//Kovarianzmatrix Rauschen
MatrixXd Noc;
Noc=CovNoi(No);
//Kovarianzmarix abhängig von Daten
MatrixXd Cy;
Cy=CovD( L, Mc, Noc);
// gewichtung für jeden Sensor für jede Quelle
MatrixXd W;
W=Weight( L, Cy);
//Gewichtetes Signal für jeden Quelle
//W MxN Y NxT
//W in Vektor 1xM*N
MatrixXd Yw;
Yw=WeightSig(W, Y,  M,  N,  T);
//addition aller sensorsignale zu einem
//es folgen  Beamsignale, für jede quelle
MatrixXd B;
B= Yw.rowwise().sum();

return 0;
}




