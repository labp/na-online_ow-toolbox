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
 * WLROIController instances for the defined DataType and FilterType template parameter.\n
 * \n
 * This class can be used in this appearance, but it also can be inherited if the derived
 * class wants to override the recalculate() method by a own implementation.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup util
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
     * \param data The data container.
     * \param branch The WRMBranch the controller branch belongs to.
     */
    WLROICtrlBranch( boost::shared_ptr< DataType > data, boost::shared_ptr< WRMBranch > branch,
                    boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner );

    /**
     * Destroys the WLROICtrlBranch.
     */
    virtual ~WLROICtrlBranch();

    /**
     * Gets the overall filter structure of the whole branch depending on all ROI controllers of the branch.
     *
     * \return Returns a shared pointer on the filter structure.
     */
    boost::shared_ptr< FilterType > getFilter();

    /**
     * Gets the WRMBranch.
     *
     * \return Returns a shared pointer on the branch.
     */
    boost::shared_ptr< WRMBranch > getBranch();

    /**
     * Determines whether or not the branches filter has to recalculate.
     *
     * \return Returns true if the branch is dated, otherwise false.
     */
    bool isDirty() const;

    /**
     * Add a new ROI controller to the branch.
     *
     * \param roi A shared pointer on the new ROI controller.
     */
    void addRoi( boost::shared_ptr< WLROIController< DataType, FilterType > > roi );

    /**
     * Gets the ROI list.
     *
     * \return Returns a list of shared pointers on ROI controllers.
     */
    std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > > getROIs();

    /**
     * Removes a roi from the branch.
     *
     * \param roi The ROI to remove.
     */
    void removeRoi( osg::ref_ptr< WROI > roi );

    /**
     * Checks if empty.
     *
     * \return Returns true when this branch contains no ROI.
     */
    bool isEmpty();

    /**
     * Sets the dirty flag.
     */
    void setDirty();

    /**
     * Sets the data.
     *
     * \param data A shared pointer on a DataType object.
     */
    void setData( boost::shared_ptr< DataType > data );

    /**
     * Gets the branches filter combiner.
     *
     * @return Returns a shared pointer on a constant WLROIFilterCombiner.
     */
    boost::shared_ptr< const WLROIFilterCombiner< FilterType > > getCombiner() const;

    /**
     * Sets the new filer combiner.
     *
     * @param combiner A shared pointer on the new filter combiner.
     */
    void setCombiner( boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner );

protected:
    boost::shared_ptr< DataType > m_data; //!< The data container.

    boost::shared_ptr< FilterType > m_filter; //!< The filter structure.

    virtual void recalculate(); //!< Recalculates the filter structure.

private:
    boost::shared_ptr< WRMBranch > m_branch; //!< The WRMBranch the controller branch belongs to.

    bool m_dirty; //!< Flag to determine the branch as dated.

    /**
     * A list of the branches ROIs (the controller of the ROIs).
     */
    std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > > m_rois;

    /**
     * Signal that can be used to update the controller branch.
     */
    boost::shared_ptr< boost::function< void() > > m_changeSignal;

    /**
     * Signal that can be used to update the controller branch.
     */
    boost::shared_ptr< boost::function< void() > > m_changeRoiSignal;

    boost::shared_ptr< WLROIFilterCombiner< FilterType > > m_combiner; //!< The filter combiner.
};

template< typename DataType, typename FilterType >
inline WLROICtrlBranch< DataType, FilterType >::WLROICtrlBranch( boost::shared_ptr< DataType > data,
                boost::shared_ptr< WRMBranch > branch, boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner ) :
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
inline boost::shared_ptr< FilterType > WLROICtrlBranch< DataType, FilterType >::getFilter()
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
inline std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > > WLROICtrlBranch< DataType, FilterType >::getROIs()
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

    for( typename std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > >::iterator it = m_rois.begin();
                    it != m_rois.end(); ++it )
    {
        ( *it )->setData( m_data );
    }

    setDirty();
}

template< typename DataType, typename FilterType >
inline boost::shared_ptr< const WLROIFilterCombiner< FilterType > > WLROICtrlBranch< DataType, FilterType >::getCombiner() const
{
    return m_combiner;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::setCombiner(
                boost::shared_ptr< WLROIFilterCombiner< FilterType > > combiner )
{
    m_combiner = combiner;
}

template< typename DataType, typename FilterType >
inline void WLROICtrlBranch< DataType, FilterType >::recalculate()
{
    m_filter.reset( new FilterType );

    if( !m_combiner )
    {
        return;
    }

    for( typename std::list< boost::shared_ptr< WLROIController< DataType, FilterType > > >::iterator it = m_rois.begin();
                    it != m_rois.end(); ++it )
    {
        m_combiner->setFilter( m_filter, ( *it )->getFilter() );
        if( m_combiner->combine() )
        {
            m_filter = m_combiner->getCombined();
        }
    }
}

#endif  // WLROICTRLBRANCH_H_
