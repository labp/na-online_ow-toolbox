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

#ifndef WLROICONTROLLER_H_
#define WLROICONTROLLER_H_

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <core/graphicsEngine/WROI.h>

/**
 * The abstract generic class WLROIController defines an interface between the graphical
 * ROI-Geode and the algorithm for filtering the data depending on the ROI configuration.\n
 * \n
 * Any derived class form WLROIController has to implement the recalculate() method and to
 * initialize the m_filter structure.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup util
 */
template< typename DataType, typename FilterType >
class WLROIController
{
public:
    /**
     * A shared pointer on a WLROIController.
     */
    typedef boost::shared_ptr< WLROIController< DataType, FilterType > > SPtr;

    /**
     * A shared pointer on a constant WLROIController.
     */
    typedef boost::shared_ptr< const WLROIController< DataType, FilterType > > ConstSPtr;

    /**
     * Constructs a new WLROIController.
     *
     * \param roi The ROI object.
     * \param data The data to calculate.
     */
    WLROIController( osg::ref_ptr< WROI > roi, boost::shared_ptr< DataType > data );

    /**
     * Destroys the WLROIController.
     */
    virtual ~WLROIController();

    /**
     * Forces the concrete controller to recalculate the filter structure.
     */
    virtual void recalculate() = 0;

    /**
     * Gets the filter structure.
     *
     * \return Returns a shared pointer on the filter structure.
     */
    boost::shared_ptr< FilterType > getFilter();

    /**
     * Gets the graphical ROI.
     *
     * \return Gets a reference pointer on a WROI.
     */
    osg::ref_ptr< WROI > getRoi();

    /**
     * Marks the controller as dated and causes a recalculate() when serving the
     * filter the next time.
     */
    void setDirty();

    /**
     * Sets the data.
     *
     * \param data A shared pointer on a DataType object.
     */
    void setData( boost::shared_ptr< DataType > data );

protected:
    osg::ref_ptr< WROI > m_roi; //!< The ROI object.

    boost::shared_ptr< DataType > m_data; //!< The data container.

    boost::shared_ptr< FilterType > m_filter; //!< The filter data structure.

    bool m_dirty; //!< Flag, to determine the filter as dated.

    boost::shared_ptr< boost::function< void() > > m_changeRoiSignal; //!< Signal for updating the controller.
};

template< typename DataType, typename FilterType >
inline WLROIController< DataType, FilterType >::WLROIController( osg::ref_ptr< WROI > roi, boost::shared_ptr< DataType > data ) :
                m_roi( roi ), m_data( data ), m_filter( boost::shared_ptr< FilterType >( new FilterType ) ), m_dirty( true )
{
    m_changeRoiSignal = boost::shared_ptr< boost::function< void() > >(
                    new boost::function< void() >( boost::bind( &WLROIController< DataType, FilterType >::setDirty, this ) ) );
    m_roi->addROIChangeNotifier( m_changeRoiSignal );
}

template< typename DataType, typename FilterType >
inline WLROIController< DataType, FilterType >::~WLROIController()
{
    m_roi->removeROIChangeNotifier( m_changeRoiSignal );
}

template< typename DataType, typename FilterType >
inline boost::shared_ptr< FilterType > WLROIController< DataType, FilterType >::getFilter()
{
    if( m_dirty )
    {
        recalculate();
    }

    return m_filter;
}

template< typename DataType, typename FilterType >
inline osg::ref_ptr< WROI > WLROIController< DataType, FilterType >::getRoi()
{
    return m_roi;
}

template< typename DataType, typename FilterType >
inline void WLROIController< DataType, FilterType >::setDirty()
{
    m_dirty = true;
}

template< typename DataType, typename FilterType >
inline void WLROIController< DataType, FilterType >::setData( boost::shared_ptr< DataType > data )
{
    m_data = data;

    setDirty();
}

#endif  // WLROICONTROLLER_H_
