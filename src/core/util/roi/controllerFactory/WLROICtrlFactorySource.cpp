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
