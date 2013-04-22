#ifndef PROJECTIONITEM_H
#define PROJECTIONITEM_H

#include <string>
#include <inttypes.h>
#include "LFArrayFloat2d.h"

/**
 * Projection Item block (314)
 */
class LFProjectionItem
{
public:
    enum proj_item_t/**< Projection (SSP) types */
    {
        pi_none = 0,/**< Not known. */
        pi_field = 1,/**< Magnetic field shape. */
        pi_dip_fix = 2,/**< Fixed (in head coordinates) position dipole */
        pi_dip_rot = 3,/**< Rotating (in head coordinates) dipole */
        pi_homog_grad = 4,/**< Homogeneous gradient */
        pi_homog_field = 5/**< Homogeneous fiels */
    };
protected:
    string m_Description;/**< Description (206) */
    proj_item_t m_Kind;/**< Type of projection definition, default == pi_none (3411) */
    int32_t m_NumberOfChannels;/**< Number of Channels, default == -1 (200) */
    string m_ChannelNameList;/**< Names of the channels of the projection vectors (3417) */
    float m_Time;/**< s. time of the field sample, default == -FLT_MAX (3412) */
    //int32_t m_NumberOfProjectionVectors;/**< Number of projection vectors (3414) */
    LFArrayFloat2d m_ProjectionVectors;/**< Projection Vectors (3415) */
public:
    /**
     * Constructor
     */
    LFProjectionItem();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns Description
     */
    string& GetDescription();
    /**
     * Returns the type of projection definition
     */
    proj_item_t GetKind();
    /**
     * Returns Number of Channels
     */
    int32_t GetNumberOfChannels();
    /**
     * Returns Channel Name List
     */
    string& GetChannelNameList();
    /**
     * Returns the time of the field sample
     */
    float GetTime();
    /**
     * Returns the number of projection vectors
     */
    int32_t GetNumberOfProjectionVectors();
    /**
     * Returns projection vectors
     */
    LFArrayFloat2d& GetProjectionVectors();
    /**
     * Sets Description
     */
    void SetDescription( const char* src );
    /**
     * Sets the type of projection definition
     */
    void SetKind( const proj_item_t src );
    /**
     * Sets Number of Channels
     */
    void SetNumberOfChannels( const int32_t src );
    /**
     * Sets Channel Name List
     */
    void SetChannelNameList( const char* src );
    /**
     * Sets the time of the field sample
     */
    void SetTime( const float src );
    /**
     * Sets the number of projection vectors
     */
    //void SetNumberOfProjectionVectors(int32_t src);
};

#endif
