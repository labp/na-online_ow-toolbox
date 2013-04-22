#ifndef LFCOORDTRANS_H
#define LFCOORDTRANS_H

#include <inttypes.h>
using namespace std;

/**
 * Coordinate transformations
 */
class LFCoordTrans
{
protected:
    int32_t m_From;/**< The source coordinate frame, default == -1 */
    int32_t m_To;/**< The destination coordinate frame, default == -1 */
    float m_Rot[3][3];/**< Rotation matrix between the coordinate frames, default == 0 */
    float m_Move[3];/**< Location of the source coordinate system origin in destination coordinates, default == 0 */
    float m_InvRot[3][3];/**< Inverse rotation matrix between the coordinate frames, default == 0 */
    float m_InvMove[3];/**< Location of the destination coordinate system origin in source coordinates, default == 0 */
public:
    /**
     * Constructor
     */
    LFCoordTrans();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the source coordinate frame
     */
    int32_t GetFrom();
    /**
     * Returns the destination coordinate frame
     */
    int32_t GetTo();
    /**
     * Returns rotation matrix between the coordinate frames
     */
    float (&GetRot())[3][3];
    /**
     * Returns location of the source coordinate system origin in destination coordinates
     */
    float* GetMove();
    /**
     * Returns inverse rotation matrix between the coordinate frames
     */
    float (&GetInvRot())[3][3];
    /**
     * Returns location of the destination coordinate system origin in source coordinates
     */
    float* GetInvMove();
    /**
     * Sets the source coordinate frame
     */
    void SetFrom( const int32_t src );
    /**
     * Sets the destination coordinate frame
     */
    void SetTo( const int32_t src );
    /**
     * Sets rotation matrix between the coordinate frames
     */
    void SetRot( const float src[3][3] );
    /**
     * Sets location of the source coordinate system origin in destination coordinates
     */
    void SetMove( const float* src );
    /**
     * Sets inverse rotation matrix between the coordinate frames
     */
    void SetInvRot( const float src[3][3] );
    /**
     * Sets location of the destination coordinate system origin in source coordinates
     */
    void SetInvMove( const float* src );
};

#endif
