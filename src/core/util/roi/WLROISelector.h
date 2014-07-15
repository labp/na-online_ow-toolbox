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

#ifndef WLROISELECTOR_H_
#define WLROISELECTOR_H_

#include <list>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WCondition.h>
#include <core/graphicsEngine/WROI.h>
#include <core/kernel/WKernel.h>
#include <core/kernel/WROIManager.h>

#include "WLROIController.h"
#include "WLROICtrlBranch.h"
#include "WLROICtrlFactory.h"

/**
 * The abstract class WLROISelector describes the interface between ROI configuration, ROI manager
 * and the used algorithm.
 * Derived classes have to initialize the @m_factory member with a concrete controller factory.
 */
template< typename DataType, typename FilterType >
class WLROISelector
{
public:

    /**
     * A shared pointer on a WLROISelector.
     */
    typedef boost::shared_ptr< WLROISelector< DataType, FilterType > > SPtr;

    /**
     * A shared pointer on a constant WLROISelector.
     */
    typedef boost::shared_ptr< const WLROISelector< DataType, FilterType > > ConstSPtr;

    /**
     * Constructs a new WLROISelector.
     *
     * @param data The data structure on which the ROIs have to applied.
     */
    WLROISelector( boost::shared_ptr< DataType > data );

    /**
     * Destroys the WLROISelector.
     */
    virtual ~WLROISelector();

    /**
     * Marks the WLROISelector as dirty and the filter structure has to recalculated.
     */
    void setDirty();

    /**
     * Gets whether the WLROISelector is dated.
     *
     * @return Returns true if the WLROISelector is dated, otherwise false.
     */
    bool getDirty() const;

    /**
     * Gets the dirty condition.
     *
     * @return Returns a shared pointer on as WCondition.
     */
    WCondition::SPtr getDirtyCondition();

    /**
     * Gets the current filter structure.
     *
     * @return Returns a shared pointer on a constant filter type.
     */
    boost::shared_ptr< const FilterType > getFilter() const;

protected:

    /**
     * Method, for recalculating the @m_filter structure for the whole ROI configuration.
     */
    virtual void recalculate() = 0;

    /**
     * Listener function for inserting ROIs.
     *
     * @param roi New ROI inserted into the ROI structure.
     */
    void slotAddRoi( osg::ref_ptr< WROI > roi );

    /**
     * Listener function for removing ROIs.
     *
     * @param roi ROI that is being removed.
     */
    void slotRemoveRoi( osg::ref_ptr< WROI > roi );

    /**
     * Listener function for removing ROIs.
     *
     * @param branch Branch that is being removed.
     */
    void slotRemoveBranch( boost::shared_ptr< WRMBranch > branch );

    /**
     * The data to calculate.
     */
    boost::shared_ptr< DataType > m_data;

    /**
     * The overall filter for the data.
     */
    boost::shared_ptr< FilterType > m_filter;

    /**
     * The ROI controller factory.
     */
    boost::shared_ptr< WLROICtrlFactory< WLROIController< DataType, FilterType >, DataType, FilterType > > m_factory;

private:

    /**
     * The dirty flag.
     */
    bool m_dirty;

    /**
     * The list of ROI controller branches.
     */
    std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > > m_branches;

    /**
     * Signal that can be used to update the selector.
     */
    boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > > m_assocRoiSignal;

    /**
     * Signal that can be used to update the selector.
     */
    boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > > m_removeRoiSignal;

    /**
     * Signal for updating the selector.
     */
    boost::shared_ptr< boost::function< void( boost::shared_ptr< WRMBranch > ) > > m_removeBranchSignal;

    /**
     * Signal that can be used to update the selector.
     */
    boost::shared_ptr< boost::function< void() > > m_changeRoiSignal;

    /**
     * Condition that fires on setDirty.
     */
    WCondition::SPtr m_dirtyCondition;
};

template< typename DataType, typename FilterType >
inline WLROISelector< DataType, FilterType >::WLROISelector( boost::shared_ptr< DataType > data ) :
                m_data( data ), m_filter( boost::shared_ptr< FilterType >( new FilterType ) ), m_dirty( true ), m_dirtyCondition(
                                boost::shared_ptr< WCondition >( new WCondition() ) )
{
    std::vector< osg::ref_ptr< WROI > > rois = WKernel::getRunningKernel()->getRoiManager()->getRois();

    m_changeRoiSignal = boost::shared_ptr< boost::function< void() > >(
                    new boost::function< void() >( boost::bind( &WLROISelector< DataType, FilterType >::setDirty, this ) ) );

    m_assocRoiSignal = boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > >(
                    new boost::function< void( osg::ref_ptr< WROI > ) >(
                                    boost::bind( &WLROISelector< DataType, FilterType >::slotAddRoi, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addAddNotifier( m_assocRoiSignal );

    m_removeRoiSignal = boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > >(
                    new boost::function< void( osg::ref_ptr< WROI > ) >(
                                    boost::bind( &WLROISelector< DataType, FilterType >::slotRemoveRoi, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addRemoveNotifier( m_removeRoiSignal );

    m_removeBranchSignal = boost::shared_ptr< boost::function< void( boost::shared_ptr< WRMBranch > ) > >(
                    new boost::function< void( boost::shared_ptr< WRMBranch > ) >(
                                    boost::bind( &WLROISelector< DataType, FilterType >::slotRemoveBranch, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addRemoveBranchNotifier( m_removeBranchSignal );

    for( size_t i = 0; i < rois.size(); ++i )
    {
        slotAddRoi( rois[i] );
        ( rois[i] )->getProperties()->getProperty( "Dirty" )->toPropBool()->set( true );
    }
}

template< typename DataType, typename FilterType >
inline WLROISelector< DataType, FilterType >::~WLROISelector()
{
    WKernel::getRunningKernel()->getRoiManager()->removeAddNotifier( m_assocRoiSignal );
    WKernel::getRunningKernel()->getRoiManager()->removeRemoveNotifier( m_removeRoiSignal );
    WKernel::getRunningKernel()->getRoiManager()->removeRemoveBranchNotifier( m_removeBranchSignal );

    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator iter = m_branches.begin();
                    iter != m_branches.end(); ++iter )
    {
        std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > > rois = ( *iter )->getROIs();
        for( typename std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > >::iterator roiIter = rois.begin();
                        roiIter != rois.end(); ++roiIter )
        {
            ( *roiIter )->getRoi()->removeROIChangeNotifier( m_changeRoiSignal );
        }
        ( *iter )->getBranch()->removeChangeNotifier( m_changeRoiSignal );
    }

    m_branches.clear();
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::setDirty()
{
    m_dirty = true;
    m_dirtyCondition->notify();
    recalculate();
}

template< typename DataType, typename FilterType >
inline bool WLROISelector< DataType, FilterType >::getDirty() const
{
    return m_dirty;
}

template< typename DataType, typename FilterType >
inline WCondition::SPtr WLROISelector< DataType, FilterType >::getDirtyCondition()
{
    return m_dirtyCondition;
}

template< typename DataType, typename FilterType >
inline boost::shared_ptr< const FilterType > WLROISelector< DataType, FilterType >::getFilter() const
{
    return m_filter;
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::slotAddRoi( osg::ref_ptr< WROI > roi )
{
    boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > branch;

    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator iter = m_branches.begin();
                    iter != m_branches.end(); ++iter )
    {
        if( ( *iter )->getBranch() == WKernel::getRunningKernel()->getRoiManager()->getBranch( roi ) )
        {
            branch = ( *iter );
        }
    }
    if( !branch )
    {
        branch = boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > >(
                        new WLROICtrlBranch< DataType, FilterType >( m_data,
                                        WKernel::getRunningKernel()->getRoiManager()->getBranch( roi ) ) );
        branch->getBranch()->addChangeNotifier( m_changeRoiSignal );
        m_branches.push_back( branch );
    }

    boost::shared_ptr< WLROIController< DataType, FilterType > > sroi( m_factory->create( "WROI", roi, m_data ) );
    branch->addRoi( sroi );
    sroi->getRoi()->addROIChangeNotifier( m_changeRoiSignal );

    setDirty();
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::slotRemoveRoi( osg::ref_ptr< WROI > roi )
{
    roi->removeROIChangeNotifier( m_changeRoiSignal );
    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator iter = m_branches.begin();
                    iter != m_branches.end(); ++iter )
    {
        ( *iter )->removeRoi( roi );

        if( ( *iter )->isEmpty() )
        {
            ( *iter )->getBranch()->removeChangeNotifier( m_changeRoiSignal );
            m_branches.erase( iter );
            break;
        }
    }
    setDirty();
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::slotRemoveBranch( boost::shared_ptr< WRMBranch > branch )
{
    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator iter = m_branches.begin();
                    iter != m_branches.end(); ++iter )
    {
        if( branch == ( *iter )->getBranch() )
        {
            // remove notifier
            branch->removeChangeNotifier( m_changeRoiSignal );
            m_branches.erase( iter );
            break;
        }
    }
    setDirty();
}

#endif /* WLROISELECTOR_H_ */
