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

#ifndef WLROISELECTORSOURCE_H_
#define WLROISELECTORSOURCE_H_

#include <list>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "WLROISelector.h"

/**
 * The WLROISelectorSource is a derivation of the abstract WLROISelector.
 * It provides the adapter between the ROI configuration, the ROI Manager of OpenWalnut
 * and the source reconstruction algorithm.
 */
class WLROISelectorSource: public WLROISelector< WLEMData, std::list< size_t > >
{

public:

    /**
     * A shared pointer on a WLROISelectorSource.
     */
    typedef boost::shared_ptr< WLROISelectorSource > SPtr;

    /**
     * A shared pointer on a constant WLROISelectorSource.
     */
    typedef boost::shared_ptr< const WLROISelectorSource > ConstSPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WLROISelectorSource.
     *
     * @param data The data container.
     */
    explicit WLROISelectorSource( WLEMData::SPtr data, WLEMDDrawable3D::SPtr drawable3D );

protected:

    /**
     * Recalculates the filter structure to select the channels includes by the ROI.
     */
    //void recalculate();

    /**
     * Event method when creating a new ROI.
     *
     * @param A reference pointer on the new ROI.
     */
    virtual void slotAddRoi( osg::ref_ptr< WROI > );

    /**
     * Event method when deleting a ROI.
     *
     * @param A reference pointer on the ROI to delete.
     */
    virtual void slotRemoveRoi( osg::ref_ptr< WROI > );

private:

    /**
     * The 3D drawable.
     */
    WLEMDDrawable3D::SPtr m_drawable3D;

};

#endif /* WLROISELECTORSOURCE_H_ */
