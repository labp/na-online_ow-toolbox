#ifndef LFBEM_H
#define LFBEM_H

#include "LFCoordTrans.h"
#include "LFBemSurface.h"
#include "LFArrayFloat2d.h"

/**
 * A boundary-element model (BEM) description (310)
 */
class LFBem
{
public:
    /**
     * Coordinate system definition
     */
    enum coord_t
    {
        c_unknown = 0/**< Unknown coordinate system */
        ,c_device = 1/**< MEG device coordinate system */
        ,c_isotrak = 2/**< 3D Digitizer coordinate system */
        ,c_hpi = 3/**< Position indicator system coordinates */
        ,c_head = 4/**< Head coordinates, Elekta Neuromag convention */
        ,c_data_volume = 5/**< Coordinate system of the data volume */
        ,c_data_slice = 6/**< Coordinates system of a slice (e.g. MRI image) */
        ,c_data_display = 7/**<  Coordinate system of the data display */
        ,c_dicom_device = 8/**< Coordinate system of the imaging device, DICOM convention */
        ,c_imaging_device = 9/**< Coord. system of the structural imaging device */
        ,c_voxel_data = 10/**< Coord. system of multidimensional voxel values */
        ,c_atlas_head = 11/**< Head coordinates of the atlas std head */
        ,c_torso = 100/**< Torso coordinates */
    };
protected:
    coord_t m_BemCoordinateFrame;/**< The coordinate frame of the model, default == c_unknown (3112) */
    LFCoordTrans m_LFCoordTrans;/**< Coordinate transformations */
    LFBemSurface m_LFBemSurface;/**< BEM surface (311) */
    LFArrayFloat2d m_BemSolutionMatrix;/**< The solution matrix (3110) */
public:
    /**
     * Constructor
     */
    LFBem();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the coordinate frame of the model (3112)
     */
    coord_t GetBemCoordinateFrame();
    /**
     * Returns coordinate transformations
     */
    LFCoordTrans& GetLFCoordTrans();
    /**
     * Returns BEM surface
     */
    LFBemSurface& GetLFBemSurface();
    /**
     * Returns the solution matrix
     */
    LFArrayFloat2d& GetBemSolutionMatrix();
    /**
     * Sets the coordinate frame of the model (3112)
     */
    void SetBemCoordinateFrame( const coord_t src );
};

#endif
