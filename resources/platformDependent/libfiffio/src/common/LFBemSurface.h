#ifndef LFBEMSURFACE_H
#define LFBEMSURFACE_H

#include <inttypes.h>

#include "LFArrayFloat3d.h"

using namespace std;

/**
 * BEM surface (311)
 */
class LFBemSurface
{
public:
    enum bem_surf_id_t/**< Surface type (bem_surf_id) (3101) */
    {
        bs_unknown2 = -1/**< Undefined */
        ,bs_unknown = 0/**< Not known */
        ,bs_brain = 1/**< Brain surface */
        ,bs_csf = 2/**< Cerospinal fluid */
        ,bs_skull = 3/**< Skull bone */
        ,bs_head = 4/**< Scalp surface */
        ,bs_blood = 11/**< Blood mass */
        ,bs_heart = 12/**< Heart surface */
        ,bs_lungs = 13/**< Lung surface */
        ,bs_torso = 14/**< Thorax surface */
        ,bs_nm122 = 21/**< NM122 sensor array */
        ,bs_unit_sphere = 22/**< Unit sphere */
        ,bs_vv = 23/**< VectorView sensor array */
    };
protected:
    bem_surf_id_t m_SurfaceId;/**< Surface type (bem_surf_id), default == bs_unknown2 (3101) */
    float m_BemSigma;/**< Conductivity of a compartment, S/m, default == -FLT_MAX (3113) */
    int32_t m_NumberOfNodes;/**< Number of nodes on a surface, default == -1 (3103) */
    int32_t m_NumberOfTriangles;/**< Number of triangles on a surface, default == -1 (3104) */
    LFArrayFloat3d m_BemSurfaceNodes;/**< Surface nodes, matrix with 3* 'Number of nodes' elements (3105) */
    LFArrayFloat3d m_BemSurfaceNormals;/**< Vertex normal unit vectors, matrix with 3* 'Number of nodes' elements (3107) */
    std::vector<int32_t> m_BemSurfaceTriangles;/**< Surface triangles, matrix with 3* 'Number of triangles' elements (3106) */
public:
    /**
     * Constructor
     */
    LFBemSurface();
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns the surface type
     */
    bem_surf_id_t GetSurfaceId();
    /**
     * Returns the conductivity of a compartment
     */
    float GetBemSigma();
    /**
     * Returns the number of nodes on a surface
     */
    int32_t GetNumberOfNodes();
    /**
     * Returns the number of triangles on a surface
     */
    int32_t GetNumberOfTriangles();
    /**
     * Returns surface nodes, matrix with 3*'Number of nodes' elements
     */
    LFArrayFloat3d& GetBemSurfaceNodes();
    /**
     * Returns vertex normal unit vectors, matrix with 3*'Number of nodes' elements
     */
    LFArrayFloat3d& GetBemSurfaceNormals();
    /**
     * Returns surface triangles, matrix with 3*'Number of triangles' elements
     */
    vector<int32_t>& GetBemSurfaceTriangles();
    /**
     * Sets the surface type
     */
    void SetSurfaceId(const bem_surf_id_t src);
    /**
     * Sets the conductivity of a compartment
     */
    void SetBemSigma(const float src);
    /**
     * Sets the number of nodes on a surface
     */
    void SetNumberOfNodes(const int32_t src);
    /**
     * Sets the number of triangles on a surface
     */
    void SetNumberOfTriangles(const int32_t src);
};

#endif  // LFBEMSURFACE_H
