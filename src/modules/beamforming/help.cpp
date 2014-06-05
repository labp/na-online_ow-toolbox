#include <iostream>
#include <fstream>
#include <limits>

std::istream& skipline( std::istream& in )
{
    return in.ignore( std::numeric_limits< std::streamsize >::max(), '\n' );
}

int main()
{
    using namespace std;
    ifstream datei("input.txt");
    if( !datei.is_open() )
    {
        cerr << "Fehler beim Oeffnen der Datei" << endl;
        return -1;
    }
    int z1, z2, z3;
    int z4, z5;
    if( datei >> skipline >> skipline >> z1 >> z2 >> z3 >> skipline >> z4 >> z5 )
    {
        // alles ohne Fehler gelesen
        cout << "Die 5. Zahl ist " << z5 << endl;
    }
    return 0;
}
