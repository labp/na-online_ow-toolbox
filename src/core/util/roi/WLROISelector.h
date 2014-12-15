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

#ifndef WLROISELECTOR_H_
#define WLROISELECTOR_H_

#include <list>
#include <typeinfo>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WCondition.h>
#include <core/common/WLogger.h>
#include <core/graphicsEngine/WROI.h>
#include <core/kernel/WKernel.h>
#include <core/kernel/WROIManager.h>

#include "controllerFactory/WLROICtrlFactory.h"
#include "filterCombiner/WLROIFilterCombiner.h"
#include "WLROIController.h"
#include "WLROICtrlBranch.h"

/**
 * The abstract class WLROISelector describes the interface between ROI configuration, ROI manager and the used algorithm.
 * Derived classes have to initialize the @m_factory member with a concrete controller factory.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup util
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
     * A shared pointer on a ROI controller factory.
     */
    typedef boost::shared_ptr< WLROICtrlFactory< WLROIController< DataType, FilterType >, DataType > > ControllerFactorySPtr;

    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructs a new WLROISelector.
     *
     * \param data The data structure on which the ROIs have to applied.
     */
    explicit WLROISelector( boost::shared_ptr< DataType > data, ControllerFactorySPtr factory,
                    boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner );

    /**
     * Destroys the WLROISelector.
     */
    virtual ~WLROISelector();

    /**
     * Marks the WLROISelector as dirty and the filter structure has to recalculated.
     */
    void setDirty();

    /**
     * Sets the data.
     *
     * \param data A shared pointer on a DataType object.
     */
    void setData( boost::shared_ptr< DataType > data );

    /**
     * Sets the factory to create new ROI controllers.
     *
     * \param factory A shared pointer on a ROI controller factory object.
     */
    void setFactory( boost::shared_ptr< ControllerFactorySPtr > factory );

    /**
     * Gets whether the WLROISelector is dated.
     *
     * \return Returns true if the WLROISelector is dated, otherwise false.
     */
    bool isDirty() const;

    /**
     * Gets the dirty condition.
     *
     * \return Returns a shared pointer on as WCondition.
     */
    WCondition::SPtr getDirtyCondition();

    /**
     * Gets the current filter structure.
     *
     * \return Returns a shared pointer on a constant filter type.
     */
    boost::shared_ptr< const FilterType > getFilter() const;

protected:
    /**
     * Method, for recalculating the @m_filter structure for the whole ROI configuration.
     */
    virtual void recalculate();

    /**
     * Listener function for inserting ROIs.
     *
     * \param roi New ROI inserted into the ROI structure.
     */
    virtual void slotAddRoi( osg::ref_ptr< WROI > );

    /**
     * Listener function for removing ROIs.
     *
     * \param roi ROI that is being removed.
     */
    virtual void slotRemoveRoi( osg::ref_ptr< WROI > roi );

    /**
     * Listener function for removing ROIs.
     *
     * \param branch Branch that is being removed.
     */
    virtual void slotRemoveBranch( boost::shared_ptr< WRMBranch > branch );

    /**
     * Generates new ROIs form the current  ROI configuration.
     */
    void generateRois();

    boost::shared_ptr< DataType > m_data; //!< The data to calculate.

    ControllerFactorySPtr m_factory; //!< The ROI controller factory.

    boost::shared_ptr< WLROIFilterCombiner< FilterType > > m_combiner; //!< The filter combiner.

    boost::shared_ptr< FilterType > m_filter; //!< The overall filter for the data.

    /**
     * The list of ROI controller branches.
     */
    std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > > m_branches;

    bool m_dirty; //!< The dirty flag.

private:
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
const std::string WLROISelector< DataType, FilterType >::CLASS = "WLROISelector";

template< typename DataType, typename FilterType >
inline WLROISelector< DataType, FilterType >::WLROISelector( boost::shared_ptr< DataType > data, ControllerFactorySPtr factory,
                boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner ) :
                m_data( data ), m_factory( factory ), m_combiner( combiner ), m_filter(
                                boost::shared_ptr< FilterType >( new FilterType ) ), m_dirty( true ), m_dirtyCondition(
                                boost::shared_ptr< WCondition >( new WCondition() ) )
{
    m_changeRoiSignal = boost::shared_ptr< boost::function< void() > >(
                    new boost::function< void() >( boost::bind( &WLROISelector< DataType, FilterType >::setDirty, this ) ) );

    m_assocRoiSignal = boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > >(
                    new boost::function< void( osg::ref_ptr< WROI > ) >(
                                    boost::bind( &WLROISelector< DataType, FilterType >::slotAddRoi, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addAddNotifier( m_assocRoiSignal );

    m_removeRoiSignal = boost::shared_ptr< boost::function< void( osg::ref_ptr< WROI > ) > >(
                    new boost::function< void( osg::ref_ptr< WROI > ) >(
                                    boost::bind( &WLROISelector::slotRemoveRoi, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addRemoveNotifier( m_removeRoiSignal );

    m_removeBranchSignal = boost::shared_ptr< boost::function< void( boost::shared_ptr< WRMBranch > ) > >(
                    new boost::function< void( boost::shared_ptr< WRMBranch > ) >(
                                    boost::bind( &WLROISelector::slotRemoveBranch, this, _1 ) ) );
    WKernel::getRunningKernel()->getRoiManager()->addRemoveBranchNotifier( m_removeBranchSignal );

    if(m_factory)
    {
        generateRois();
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
    recalculate();
    m_dirtyCondition->notify();
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::setData( boost::shared_ptr< DataType > data )
{
    m_data = data;

    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator it = m_branches.begin();
                    it != m_branches.end(); ++it )
    {
        ( *it )->setData( m_data );
    }

    setDirty();
}

template< typename DataType, typename FilterType >
inline bool WLROISelector< DataType, FilterType >::isDirty() const
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
    if( !m_factory ) // no controller factory was configured.
    {
        return;
    }

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
                                        WKernel::getRunningKernel()->getRoiManager()->getBranch( roi ), m_combiner ) );
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

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::recalculate()
{
    if( !m_dirty )
    {
        return;
    }

    m_filter.reset( new FilterType );

    if( !m_combiner )
    {
        return;
    }

    if( !m_data )
    {
        return;
    }

    wlog::debug( CLASS ) << "recalculate()";

    for( typename std::list< boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > >::iterator it = m_branches.begin();
                    it != m_branches.end(); ++it )
    {
        m_combiner->setFilter( m_filter, ( *it )->getFilter() );
        if( m_combiner->combine() )
        {
            m_filter = m_combiner->getCombined();
        }
    }

    m_dirty = false;
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::setFactory( boost::shared_ptr< ControllerFactorySPtr > factory )
{
    m_factory = factory;
}

template< typename DataType, typename FilterType >
inline void WLROISelector< DataType, FilterType >::generateRois()
{
    std::vector< osg::ref_ptr< WROI > > rois = WKernel::getRunningKernel()->getRoiManager()->getRois();

    for( size_t i = 0; i < rois.size(); ++i )
    {
        slotAddRoi( rois[i] );
        ( rois[i] )->getProperties()->getProperty( "Dirty" )->toPropBool()->set( true );
    }
}

#endif  // WLROISELECTOR_H_
