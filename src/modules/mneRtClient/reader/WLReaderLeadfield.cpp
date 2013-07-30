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

#include <mne/mne_forwardsolution.h>
#include <QFile>

#include <core/common/WLogger.h>

#include "WLReaderLeadfield.h"

using std::string;

const string WLReaderLeadfield::CLASS = "WLReaderLeadfield";

WLReaderLeadfield::WLReaderLeadfield( string fname ) throw( WDHNoSuchFile ) :
                WReader( fname )
{
}

WLReaderLeadfield::~WLReaderLeadfield()
{
}

bool WLReaderLeadfield::read( WLMatrix::SPtr& leadfield )
{
    QFile fileIn( m_fname.c_str() );

    MNELIB::MNEForwardSolution::SPtr fwdSolution( new MNELIB::MNEForwardSolution( fileIn ) );
    if( fwdSolution->isEmpty() )
    {
        wlog::error( CLASS ) << "Could not read leadfield file!";
        return false;
    }

#ifdef LABP_FLOAT_COMPUTATION
    WLMatrix::MatrixT* matrix = new WLMatrix::MatrixT(fwdSolution->sol->data.cast<ScalarT>());
#else
    WLMatrix::MatrixT* matrix = new WLMatrix::MatrixT( fwdSolution->sol->data );
#endif  // LABP_FLOAT_COMPUTATION
    leadfield.reset( matrix );
    wlog::info( CLASS ) << "Matrix size: " << leadfield->rows() << "x" << leadfield->cols();
    return true;
}
