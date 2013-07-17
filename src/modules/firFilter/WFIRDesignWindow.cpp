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

#include <cmath>

#include <core/common/WAssert.h>

#include "WFIRFilter.h"
#include "WFIRDesignWindow.h"

WFIRDesignWindow::WFIRDesignWindow()
{
}

WFIRDesignWindow::~WFIRDesignWindow()
{
}

WFIRFilter::ScalarT WFIRDesignWindow::getFactor( WFIRFilter::WEWindowsType::Enum windowtype, int order, int n )
{
    switch( windowtype )
    {
        case WFIRFilter::WEWindowsType::HAMMING:
            return 0.54 - ( 0.46 * cos( ( 2.0 * M_PI * n ) / order ) );
        case WFIRFilter::WEWindowsType::BARLETT:
            return 1.0 - ( ( 2.0 * fabs( n - ( ( order ) / 2.0 ) ) ) / ( order ) );
        case WFIRFilter::WEWindowsType::BLACKMAN:
            return 0.42 - ( 0.5 * cos( ( 2.0 * M_PI * n ) / ( order ) ) ) + 0.08 * cos( ( 4.0 * M_PI * n ) / ( order ) );
        case WFIRFilter::WEWindowsType::HANNING:
            return 0.5 * ( 1 - cos( ( 2 * M_PI * n ) / ( order ) ) );
        case WFIRFilter::WEWindowsType::RECTANGLE:
            return 1.0;
        default:
            WAssert( false, "Unknown WEWindowsType!" );
            return 1.0; // rectangle window
    }
}
