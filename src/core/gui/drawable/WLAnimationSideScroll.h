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

#ifndef WLANIMATIONSIDESCROLL_H_
#define WLANIMATIONSIDESCROLL_H_

#include <list>
#include <string>

#include <osg/Group>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec2d>
#include <osg/Vec3d>

#include <core/common/WRealtimeTimer.h>

#include "core/data/WLEMMeasurement.h"

/**
 * A side scroll animation from right to left. The osg::nodes are moved horizontal by a 2D translation.
 * First node is positioned to the defined start position. New nodes are appended to the right end of the previous node.
 *
 * \author pieloth
 * \ingroup gui
 */
class WLAnimationSideScroll
{
public:
    static const std::string CLASS;

    class EMMNode: public osg::PositionAttitudeTransform
    {
    public:
        explicit EMMNode( WLEMMeasurement::SPtr emm );

        ~EMMNode();

        WLEMMeasurement::SPtr getEmm();

    private:
        WLEMMeasurement::SPtr m_emm;
    };

    explicit WLAnimationSideScroll( osg::ref_ptr< osg::Group > groupPAT );
    virtual ~WLAnimationSideScroll();

    /**
     * Gets the x-part of the translation vector. This value is equivalent to the vector length
     *
     * \return x-part of the translation vector
     */
    double getXTranslation() const;

    /**
     * Sets the x-part of the translation vector.
     *
     * \param vec
     */
    void setXTranslation( double vec );

    /**
     * Gets the horizontal length of a block aka osg::node.
     *
     * \return block length in pixel
     */
    double getXBlockLength() const;

    /**
     * Sets the horizontal block length.
     *
     * \param length Length in pixel
     */
    void setXBlockLength( double length );

    /**
     * Gets the initial start position of a node.
     *
     * \return start position
     */
    osg::Vec2d getStartPosition() const;

    /**
     * Sets the start position of a node.
     *
     * \param vec Start position
     */
    void setStartPosition( const osg::Vec2d& vec );

    /**
     * Gets the time for a complete translation.
     *
     * \return Time in seconds
     */
    double getTime() const;

    /**
     * Sets the time for a complete translation.
     *
     * \param sec Time in seconds
     */
    void setTime( double sec );

    /**
     * Gets the upper frame limit.
     *
     * \return max frames per second
     */
    int getMaxFps() const;

    /**
     * Sets a frame limit to reduce computational effort.
     *
     * \param maxFps max. frames per second
     */
    void setMaxFps( int maxFps );

    /**
     * Appends a new node to the animation.
     *
     * \param node Node to append
     */
    void append( osg::ref_ptr< EMMNode > node );

    /**
     * Moves all nodes depending on the elapsed time period.
     */
    void sweep();

    /**
     * Toggle start/stop animation.
     * \param pause true if animation shall stop, false to start the animation.
     */
    void setPause( bool pause );

    /**
     * Indicates if the animation is paused.
     *
     * \return true if animation is stopped
     */
    bool isPaused() const;

    /**
     * Gets the synchronization threshold.
     *
     * \return threshold in pixel
     */
    double getSyncThreshold() const;

    /**
     * Sets the synchronization threshold.
     * If the starting position of a new node differs by <threshold> pixel of the initial start position, the initial start position is used instead.
     *
     * \param threshold synchronization threshold in pixel
     */
    void setSyncThreshold( double threshold );

    const std::list< osg::ref_ptr< EMMNode > >& getNodes() const;

private:
    osg::Vec3d m_translation;

    osg::Vec3d m_startPosition;

    osg::Vec3d m_blockLength;

    bool m_pause;

    double m_time;

    double m_syncThreshold;

    double m_maxFps;

    std::list< osg::ref_ptr< EMMNode > > m_listPAT;

    osg::ref_ptr< osg::Group > m_groupPAT;

    WRealtimeTimer m_fpsTimer;
};

#endif  // WLANIMATIONSIDESCROLL_H_
