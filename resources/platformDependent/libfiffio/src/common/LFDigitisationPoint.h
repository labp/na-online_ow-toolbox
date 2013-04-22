#ifndef LFDIGITISATIONPOINT_H
#define LFDIGITISATIONPOINT_H

#include <inttypes.h>

#include "LFArrayFloat3d.h"

/**
 * Digitisation Point (dig_point_t structure)
 */
class LFDigitisationPoint
{
protected:
    int32_t m_Kind;/**< What kind of points these are, default == -1 */
    int32_t m_Ident;/**< Running number for this point set, default == -1 */
    LFArrayFloat3d m_Rr;/**< Locations of points in meters (float[3][n]) */
public:
    /**
     * Constructor
     */
    LFDigitisationPoint();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the kind of points
     */
    int32_t GetKind();
    /**
     * Returns the running number for this point set
     */
    int32_t GetIdent();
    /**
     * Returns locations of points in meters
     */
    LFArrayFloat3d& GetRr();
    /**
     * Sets the kind of points
     */
    void SetKind( const int32_t src );
    /**
     * Sets the running number for this point set
     */
    void SetIdent( const int32_t src );
};

#endif
