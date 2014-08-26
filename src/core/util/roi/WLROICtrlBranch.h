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

#ifndef WLROICTRLBRANCH_H_
#define WLROICTRLBRANCH_H_

#include <list>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <core/kernel/WRMBranch.h>

#include "filterCombiner/WLROIFilterCombiner.h"
#include "WLROIController.h"

/**
 * The generic class WLROICtrlBranch provides an interface for collection and grouping
 * WLROIController instances for the defined DataType and FilterType template parameter.
 *
 * This class can be used in this appearance, but it also can be inherited if the derived
 * class wants to override the recalculate() method by a own implementation.
 */
template< typename DataType, typename FilterType >
class WLROICtrlBranch
{
public:

    /**
     * A shared pointer on a WLROICtrlBranch.
     */
    typedef boost::shared_ptr< WLROICtrlBranch< DataType, FilterType > > SPtr;

    /**
     * A shared pointer on a constant WLROICtrlBranch.
     */
    typedef boost::shared_ptr< const WLROICtrlBranch< DataType, FilterType > > ConstSPtr;

    /**
     * Constructs a new WLROICtrlBranch.
     *
     * @param data The data container.
     * @param branch The WRMBranch the controller branch belongs to.
     */
    WLROICtrlBranch( typename WLROIController< DataType, FilterType >::DataTypeSPtr data, boost::shared_ptr< WRMBranch > branch,
                    WLROIFilterCombiner::SPtr combiner );

    /**
     * Destroys the WLROICtrlBranch.
     */
    virtual ~WLROICtrlBranch();

    /**
     * Gets the overall filter structure of the whole branch depending on all ROI controllers of the branch.
     *
     * @return Returns a shared pointer on the filter structure.
     */
    typename WLROIController< DataType, FilterType >::FilterTypeSPtr getFilter();

    /**
     * Gets the WRMBranch.
     *
     * @return Returns a shared pointer on the branch.
     */
    boost::shared_ptr< WRMBranch > getBranch();

    /**
     * Determines whether or not the branches filter has to recalculate.
     *
     * @return Returns true if the branch is dated, otherwise false.
     */
    bool isDirty() const;

    /**
     * Add a new ROI controller to the branch.
     *
     * @param roi A shared pointer on the new ROI controller.
     */
    void addRoi( boost::shared_ptr< WLROIController< DataType, FilterType > > roi );

    /**
     * Gets the ROI list.
     *
     * @return Returns a list of shared pointers on ROI controllers.
     */
    std::list< typename WLROIController< DataType, FilterType >::SPtr > getROIs();

    /**
     * Removes a roi from the branch.
     *
     * @param roi The ROI to remove.
     */
    void removeRoi( osg::ref_ptr< WROI > roi );

    /**
     * Checks if empty.
     *
     * @return Returns true when this branch contains no ROI.
     */
    bool isEmpty();

    /**
     * Sets the dirty flag.
     */
    void setDirty();

    /**
     * Sets the data.
     *
     * @param data A shared pointer on a DataType object.
     */
    void setData( boost::shared_ptr< DataType > data );

    /**
     * Gets the branches filter combiner.
     *
     * @return Returns a shared pointer on a constant WLROIFilterCombiner.
     */
    WLROIFilterCombiner::ConstSPtr getCombiner() const;

    /**
     * Sets the new filer combiner.
     *
     * @param combiner A shared pointer on the new filter combiner.
     */
    void setCombiner( WLROIFilterCombiner::SPtr combiner );

protected:

    /**
     * The data container.
     */
    typename WLROIController< DataType, FilterType >::DataTypeSPtr m_data;

    /**
     * The filter structure.
     */
    typename WLROIController< DataType, FilterType >::FilterTypeSPtr m_filter;

    /**
     * Recalculates the filter structure.
     */
    virtual void recalculate();

private:

    /**
     * The WRMBranch the controller branch belongs to.
     */
    boost::shared_ptr< WRMBranch > m_branch;

    /**
     * Flag to determine the branch as dated.
     */
    bool m_dirty;

    /**
     * A list of the branches ROIs (the controller of the ROIs).
     */
    std::list< typename WLROIController< DataType, FilterType >::SPtr > m_rois;

    /**
     * Signal that can be used to update the controller branch.
     */
    boost::shared_ptr< boost::function< void() > > m_changeSignal;

    /**
     * Signal that can be used to update the controller branch.
     */
    boost::shared_ptr< boost::function< void() > > m_changeRoiSignal;

    /**
     * The filter combiner.
     */
    WLROIFilterCombiner::SPtr m_combiner;
};

template< typename DataType, typename FilterType >
inline WLROICtrlBranch< DataType, FilterType >::WLROICtrlBranch(
                typename WLROIController< DataType, FilterType >::DataTypeSPtr data, boost::shared_ptr< WRMBranch > branch,
                WLROIFilterCombiner::SPtr combiner ) :
                m_data( data ), m_filter( boost::shared_ptr< FilterType >( new FilterType ) ), m_branch( branch ), m_dirty(
                                true ), m_combiner( combiner )
{
    m_changeSignal = boost::shared_ptr< boost::function< void() > >(
                    new boost::function< void() >( boost::bind( &WLROICtrlBranch< DataType, FilterType >::setDirty, this ) ) );
    m_branch->addChangeNotifier( m_changeSignal );

    m_changeRoiSignal = boost::shared_ptr< boost::function< void() > >(
                    new boost::function< void() >( boost::bind( &WLROICtrlBranch< DataType, FilterType >::setDirty, this ) ) );
}

template< typename DataType, typename FilterType >
inline WLROICtrlBranch< DataType, FilterType >::~WLROICtrlBranch()
{
    m_branch->removeChangeNotifier( m_changeSignal );

    // We need the following because not all ROIs are removed per slot below
    for( typename std::list< typename WLROIController< DataType, FilterType >::SPtr >::iterator roiIter = m_rois.begin();
                    roiIter != m_rois.end(); ++roiIter )
    {
        ( *roiIter )->getRoi()->removeROIChangeNotifier( m_changeRoiSignal );
    }
}

template< typename DataType, typename FilterType >
inline typename WLROIController< DataType, FilterType >::FilterTypeSPtr WLROICtrlBranch< DataType, FilterType >::getFilter()
{
    if( m_dirty )
    {
        recalculate();
    }

    return m_filter;
}

template< typename DataType, typename FilterType >
inline boost::shared_ptr< WRMBranch > WLROICtrlBranch< DataType, FilterType >::getBranch()
{
    return m_branch;
}

template< typename DataType, typename FilterType >
inline bool WLROICtrlBranch< DataType, FilterType >::isDirty() const
{
    return m_dirty;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::addRoi( boost::shared_ptr< WLROIController< DataType, FilterType > > roi )
{
    m_rois.push_back( roi );
    roi->getRoi()->addROIChangeNotifier( m_changeRoiSignal );
}

template< typename DataType, typename FilterType >
inline std::list< typename WLROIController< DataType, FilterType >::SPtr > WLROICtrlBranch< DataType, FilterType >::getROIs()
{
    return m_rois;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::removeRoi( osg::ref_ptr< WROI > roi )
{
    for( typename std::list< typename WLROIController< DataType, FilterType >::SPtr >::iterator iter = m_rois.begin();
                    iter != m_rois.end(); ++iter )
    {
        if( ( *iter )->getRoi() == roi )
        {
            ( *iter )->getRoi()->removeROIChangeNotifier( m_changeRoiSignal );
            m_rois.erase( iter );
            break;
        }
    }
}

template< typename DataType, typename FilterType >
inline bool WLROICtrlBranch< DataType, FilterType >::isEmpty()
{
    return m_rois.empty();
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::setDirty()
{
    m_dirty = true;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::setData( boost::shared_ptr< DataType > data )
{
    m_data = data;

    boost::shared_ptr< WLROIController< DataType, FilterType > > controller;
    BOOST_FOREACH(controller, m_rois)
    {
        controller->setData( m_data );
    }

    setDirty();
}

template< typename DataType, typename FilterType >
inline WLROIFilterCombiner::ConstSPtr WLROICtrlBranch< DataType, FilterType >::getCombiner() const
{
    return m_combiner;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::setCombiner( WLROIFilterCombiner::SPtr combiner )
{
    m_combiner = combiner;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::recalculate()
{
    if( !m_combiner )
    {
        return;
    }

    m_filter.reset( new FilterType );

    boost::shared_ptr< WLROIController< DataType, FilterType > > controller;

    BOOST_FOREACH(controller, m_rois)
    {
        m_combiner->setFilter< FilterType >( m_filter, controller->getFilter() );
        m_filter = m_combiner->getFilter< FilterType >();
    }
}

#endif /* WLROICTRLBRANCH_H_ */
