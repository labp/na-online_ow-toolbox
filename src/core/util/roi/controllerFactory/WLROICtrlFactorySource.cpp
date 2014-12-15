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

#include "WLROICtrlCreatorImpl.h"
#include "WLROICtrlFactorySource.h"

WLROICtrlFactorySource::WLROICtrlFactorySource()
{
}

WLROIController< WLEMMSurface, std::list< size_t > >* WLROICtrlFactorySource::create( const std::string& name,
                osg::ref_ptr< WROI > roi, boost::shared_ptr< WLEMMSurface > data ) const
{
    citerTc it = find( name );
    if( it != end() && it->second && data )
    {
        return it->second->create( roi, data );
    }
    else
    {
        return new WLROIControllerSource( roi, data );
    }
}
