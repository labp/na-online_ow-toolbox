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

#ifndef WLMARKTIMEPOSITIONHANDLER_H_
#define WLMARKTIMEPOSITIONHANDLER_H_

#include <string>

#include <osg/ref_ptr>

#include <core/ui/WUIViewEventHandler.h>

#include <core/kernel/WModuleOutputData.h>

#include "core/data/WLEMMCommand.h"
#include "core/gui/drawable/WLEMDDrawable2D.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"

/**
 * Marks the 2D view at the position, informs the 3D view and connected modules about the new time position.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLMarkTimePositionHandler: public WUIViewEventHandler
{
public:
    /**
     * Abbreviation for a osg::ref_ptr on a instance of this class.
     */
    typedef osg::ref_ptr< WLMarkTimePositionHandler > RefPtr;

    static const std::string CLASS;

    WLMarkTimePositionHandler( WLEMDDrawable2D::SPtr initiator, WLEMDDrawable3D::SPtr acceptor,
                    WModuleOutputData< WLEMMCommand >::SPtr output );

    virtual ~WLMarkTimePositionHandler();

    void setDrawables( WLEMDDrawable2D::SPtr drawable2D, WLEMDDrawable3D::SPtr drawable3D );

    virtual void handleDrag( WVector2f mousePos, int buttonMask );

    virtual void handlePush( WVector2f mousePos, int button );

private:
    WLEMDDrawable2D::SPtr m_initiator;
    WLEMDDrawable3D::SPtr m_acceptor;
    WModuleOutputData< WLEMMCommand >::SPtr m_output;
};

#endif  // WLMARKTIMEPOSITIONHANDLER_H_
