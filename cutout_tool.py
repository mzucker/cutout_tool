import sys
import os
import argparse
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import svgelements
import triangle as tr
import shapely.geometry as sgeom
import numpy as np

# note: all internal length units are mm!

BBOX_MARGIN = 5.0 #mm

DEFAULT_CUTTER_DIAMETER = 45.0 # mm
DEFAULT_CUTTER_THICKNESS = 1.0 # mm
DEFAULT_CUTTER_OVERLAP = 0.5 # mm

DEFAULT_BASE_THICKNESS = 1.0 # mm
DEFAULT_RIM_WIDTH = 2.0 # mm
DEFAULT_RIM_HEIGHT = 5.0 # mm

INCH = 25.4 # mm/in

UNITS = dict(mm=1.0,
             cm=10.0,
             pt=INCH/72,
             px=INCH/96)

UNITS['in'] = INCH

UNIT_REGEXP = re.compile(r'^([0-9]+(\.[0-9]*)?)\s*([a-z][a-z])?$')

EPS = 1e-5 # small distance in mm

# Binary format for reading/writing binary STL files with numpy arrays
# see https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html
STL_DTYPE = np.dtype([('n', 'f4', 3),
                      ('xyz', 'f4', (3, 3)),
                      ('attr', 'u2')])

######################################################################
# parse a length from the command line

def length(lstr):

    m = UNIT_REGEXP.match(lstr)

    if not m:
        raise ValueError('invalid length: ' + lstr)

    quantity = float(m.group(1))
    unit = m.group(3)

    if unit is None:
        unit = 'mm'
    elif unit not in UNITS:
        raise ValueError('invalid unit:', unit)

    mm = quantity * UNITS[unit]

    return mm

######################################################################    
# parse all of the command line options

def parse_cmdline():

    parser = argparse.ArgumentParser(
        description='make 3D-printable STL cutout templates from SVG files')
    
    parser.add_argument('svgfile', metavar='SVGFILE', type=argparse.FileType('r'))

    parser.add_argument('--rim-height', '-H', metavar='LENGTH', type=length,
                        default=DEFAULT_RIM_HEIGHT,
                        help='height of rim around base (default {} mm)'.format(DEFAULT_RIM_HEIGHT))

    parser.add_argument('--rim-width', '-w', metavar='LENGTH', type=length,
                        default=DEFAULT_RIM_WIDTH,
                        help='width of rim around base (default {} mm)'.format(DEFAULT_RIM_WIDTH))

    parser.add_argument('--base-thickness', '-t', metavar='LENGTH', type=length,
                        default=DEFAULT_BASE_THICKNESS,
                        help='thickness of base (default {} mm)'.format(DEFAULT_BASE_THICKNESS))

    parser.add_argument('--cutter-diameter', '-d', metavar='LENGTH', type=length,
                        default=DEFAULT_CUTTER_DIAMETER,
                        help='diameter of cutter (default {} mm)'.format(DEFAULT_CUTTER_DIAMETER))

    parser.add_argument('--cutter-thickness', '-c', metavar='LENGTH', type=length,
                        default=DEFAULT_CUTTER_THICKNESS,
                        help='thickness of cutter (default {} mm)'.format(DEFAULT_CUTTER_THICKNESS))
    
    parser.add_argument('--cutter-overlap', '-o', metavar='LENGTH', type=length,
                        default=DEFAULT_CUTTER_OVERLAP,
                        help='overlap of corner cuts (default {} mm)'.format(DEFAULT_CUTTER_OVERLAP))

    opts = parser.parse_args()

    if opts.rim_height <= opts.base_thickness:
        raise RuntimeError('error: rim height must be >= base thickness')

    return opts

######################################################################
# check two points close enough

def points_close(p1, p2):
    return np.abs(p2-p1).max() < EPS


######################################################################
# join disjoint subpaths

def join_subpaths(contours):

    vertices = []
    connections = []

    for cidx, (contour, is_closed) in enumerate(contours):

        if is_closed:
            print('warning: ignoring unclosed sub-paths')
            return contour

        for which_end in [0, 1]:

            v_new = contour[-which_end]
            new_idx = -1

            for vidx, v_old in enumerate(vertices):
                if points_close(v_new, v_old):
                    new_idx = vidx
                    break

            if new_idx == -1:
                new_idx = len(vertices)
                vertices.append(v_new)
                connections.append([])

            connections[new_idx].append((cidx, which_end))

    sequence = -np.ones((len(contours), 2, 2), dtype=int)

    for cinfo in connections:
        if len(cinfo) != 2:
            raise RuntimeError('subpaths do not form simple chain')
        cidx0, end0 = cinfo[0]
        cidx1, end1 = cinfo[1]
        sequence[cidx0, end0] = (cidx1, end1)
        sequence[cidx1, end1] = (cidx0, end0)
        
    if np.any(sequence == -1):
        raise RuntimeError('subpaths do not form simple chain')

    mega_contour = []

    cur_contour = 0
    cur_start = 0

    for cidx in range(len(contours)):

        c = contours[cur_contour][0]

        next_contour, next_start = sequence[cur_contour, 1-cur_start]

        assert np.all(sequence[next_contour, next_start] ==
                      [cur_contour, 1-cur_start])
        
        if cur_start == 1:
            c = contours[::-1]

        if len(mega_contour):
            assert points_close(mega_contour[-1][-1], c[0])

        mega_contour.append(c[1:])

        cur_contour = next_contour
        cur_start = next_start

    return np.vstack(tuple(mega_contour))

######################################################################
# get a single path from a SVG file

def get_path(svgfile):

    svg = svgelements.SVG.parse(svgfile, reify=True, ppi=UNITS['in'])

    print('parsed', svgfile.name)
    print('viewbox:', svg.viewbox)

    w = svg.viewbox.width
    h = svg.viewbox.height

    segments = None

    for element in svg.elements():
        if (isinstance(element, svgelements.Path) or
            isinstance(element, svgelements.Polygon) or
            isinstance(element, svgelements.Rect)):
            segments = element.segments()
            break

    if segments is None:
        raise RuntimeError('no path/polygon found in SVG!')

    contours = []
    cur_contour = None
    cur_move = None

    for segment in segments:
        if isinstance(segment, svgelements.Move):
            if cur_contour is not None:
                contours.append((np.array(cur_contour), False))
                cur_contour = None
            cur_move = [ np.array(segment.end) ]
        elif isinstance(segment, svgelements.Line):
            if cur_contour is None and cur_move is None:
                raise RuntimeError('Line without previous Move!')
            elif cur_contour is None:
                cur_contour = cur_move
                cur_move = None
            assert points_close(np.array(segment.start), cur_contour[-1])
            cur_contour.append(np.array(segment.end))
        elif isinstance(segment, svgelements.Close):
            if cur_contour is None:
                raise RuntimeError('Close without previous Line!')
            assert points_close(np.array(segment.start), cur_contour[-1])
            assert points_close(np.array(segment.end), cur_contour[0])
            contours.append((np.array(cur_contour), True))
            cur_contour = None
            cur_move = None
        else:
            raise RuntimeError('invalid svg - not a simple polygon path!')

    if cur_contour is not None:
        contours.append((np.array(cur_contour), False))

    if not contours:
        raise RuntimeError('empty path!')

    if len(contours) > 1:

        contour = join_subpaths(contours)

    else:
        
        contour, is_closed = contours[0]
    
        if not is_closed:
            raise RuntimeError('contour is not closed')

    mid = 0.5*(contour.max(axis=0) + contour.min(axis=0))
    contour -= mid
    contour *= [1.0, -1.0]

    if not sgeom.LinearRing(contour).is_ccw:
        contour = contour[::-1]

    return contour

######################################################################
# get a shape to remove the cutter from inside corners
        
def get_cutter_profile(opts):

    t = opts.cutter_thickness
    r = 0.5*opts.cutter_diameter
    o = opts.cutter_overlap

    h = opts.rim_height
    
    if h >= r:
        d = r
    else:
        d = np.sqrt(r**2 - (r-h)**2)

    cutter_profile = np.array([
        [0, 0],
        [d+o, 0],
        [d+o, t],
        [0, t]
    ])

    return cutter_profile

######################################################################
# return true if corner pA pB pC is convex (wound ccw)

def is_convex(pA, pB, pC):
    x0, y0 = pB-pA
    x1, y1 = pC-pA
    return x0*y1 - x1*y0 > 0

######################################################################
# make sure that polygon has no small edges

def validate_polygon(polygon):

    idx = np.arange(len(polygon))

    diffs = polygon[idx-1] - polygon[idx]
    lengths = np.linalg.norm(diffs, axis=1)
    
    assert np.min(lengths) > EPS

######################################################################
# make a cutting polygon for a particular edge by replicating
# the cutter profile at each vertex

def make_cutting_polygon(cutter_profile, p0, p1):

    tangent = p1 - p0
    tangent /= np.linalg.norm(tangent)

    tx, ty = tangent
    normal = np.array([-ty, tx])

    R0 = np.array([-tangent, -normal])
    R1 = np.array([tangent, -normal])

    c0 = np.dot(cutter_profile, R0) + p0
    c1 = np.dot(cutter_profile, R1) + p1
    
    poly = np.vstack((c1, c0[::-1]))

    validate_polygon(poly)
    
    return poly


######################################################################
# get all cutting polygons for a given input contour

def get_cutting_polygons(contour, cutter_profile):

    cutting_polygons = []

    corners = np.zeros(len(contour), dtype=bool)

    for idx, pC in enumerate(contour):
        pA = contour[idx-2]
        pB = contour[idx-1]
        if not is_convex(pA, pB, pC):
            corners[idx-1] = True

    for idx, pB in enumerate(contour):
        pA = contour[idx-1]
        if corners[idx] or corners[idx-1]:
            cutting_polygons.append(make_cutting_polygon(cutter_profile, pA, pB))
        
    return cutting_polygons

######################################################################
# convert shapely polygon to n-by-2 numpy array

def get_coords_as_array(poly):
    return np.array(poly.exterior)

######################################################################
# tries to find a pair of points:
#
#   - one outside of container
#   - one inside of container but not inside of excluder 
#
# note excluder can be none

def get_label_point(container, excluder, edge_dist):

    carray = get_coords_as_array(container)

    p_inside = None
    p_outside = None

    for idx, p1 in enumerate(carray):
        p0 = carray[idx-1]

        tangent = p1 - p0
        tlen = np.linalg.norm(tangent)

        if tlen < 1e-5:
            continue

        tangent /= tlen
        tx, ty = tangent
        normal = np.array([-ty, tx])

        candidate = sgeom.Point(0.5*(p0 + p1) - edge_dist*normal)

        if (p_inside is None and
            container.contains(candidate) and
            (excluder is None or not excluder.contains(candidate))):
            p_inside = candidate

        candidate = sgeom.Point(0.5*(p0 + p1) + edge_dist*normal)

        if (p_outside is None and 
            not container.contains(candidate)):
            p_outside = candidate

    return p_outside, p_inside

######################################################################
# get region labeling points for triangulation

def get_label_points(cpoly, cbuf, edge_dist):

    pout, ppoly = get_label_point(cpoly, cbuf, edge_dist)
    _, pbuf = get_label_point(cbuf, None, edge_dist)

    return pout, ppoly, pbuf

######################################################################
# get line segment indices for a polygon

def make_segments(cnt, offset):
    idx = np.arange(cnt)
    idx = np.hstack( (idx.reshape(cnt, 1),
                      ((idx + 1) % cnt).reshape(cnt, 1) ) )
    return idx + offset

######################################################################
# get line segment indices for a polygon

def cleanup_small_edges(polygon):

    new_polygon = []
    
    new_polygon = [polygon[0]]
    
    for p in polygon[1:]:
        if np.linalg.norm(p - new_polygon[-1]) > EPS:
            new_polygon.append(p)

    while np.linalg.norm(new_polygon[0]-new_polygon[-1]) < EPS:
        new_polygon = new_polygon[:-1]
        
    return np.array(new_polygon)
    
######################################################################
# plot a polygon in matplotlib

def plot_polygon(polygon, *args, **kwargs):

    if isinstance(polygon, np.ndarray):
        polygons = [polygon]
    elif isinstance(polygon, sgeom.Polygon):
        polygons = [get_coords_as_array(polygon)]
    elif isinstance(polygon, sgeom.MultiPolygon):
        polygons = [ get_coords_as_array(g) for g in polygon.geoms ]

    for polygon in polygons:
        idx = np.arange(len(polygon)+1)
        idx[-1] = 0
        plt.plot(polygon[idx,0], polygon[idx,1], *args, **kwargs)

######################################################################
# compute triangle normals

def compute_normals(vertices, triangles):

    A = vertices[triangles[:, 0]]
    B = vertices[triangles[:, 1]]
    C = vertices[triangles[:, 2]]

    u = B - A # v1 - v0 for every triangle
    v = C - A # v2 - v0 for every tringle

    # un-normalized normal vectors shape (ntris, 3)
    w = np.cross(u, v, axis=1) 

    return w / np.linalg.norm(w, axis=1).reshape(-1, 1)

######################################################################
# helper function to write STL in ASCII format.
# don't call this directly -- use write_stl() below

def _write_stl_ascii(filename, stldata):

    ostr = open(filename, 'w')

    ostr.write('solid {}\n'.format(filename))

    for n, tri in zip(stldata['n'], stldata['xyz']):

        ostr.write('  facet normal {:e} {:e} {:e}\n'.format(*n))
        ostr.write('    outer loop\n')
        for v in tri:
            ostr.write('      vertex {:e} {:e} {:e}\n'.format(*v))
        ostr.write('    endloop\n')
        ostr.write('  endfacet\n')
        
    ostr.write('endsolid\n')

######################################################################
# helper function to write STL in binary format.
# don't call this directly -- use write_stl() below

def _write_stl_binary(filename, stldata):

    ostr = open(filename, 'wb')

    header = b'BINARY STL ' + filename.encode('ascii')
    if len(header) >= 80:
        header = header[:80]
    else:
        header += b' '*(80-len(header))

    ostr.write(header)

    np.uint32(len(stldata)).tofile(ostr)
    stldata.tofile(ostr)

######################################################################
# write an STL file from given vertices and triangles
#
# vertices should be of shape (nverts, 3) and datatype float32
# triangles should be of shape (ntris 3) and datatype int

def write_stl(filename, vertices, triangles, fmt):

    assert fmt in ('ascii', 'binary')

    ntris = len(triangles)

    stldata = np.empty(ntris, dtype=STL_DTYPE)

    stldata['attr'] = 0

    stldata['xyz'] = vertices[triangles]
    triangle_vertices = stldata['xyz']

    assert triangle_vertices.shape == (ntris, 3, 3)

    # normalize 
    stldata['n'] = compute_normals(vertices, triangles)

    # call one of the helper functions above
    if fmt == 'ascii':
        _write_stl_ascii(filename, stldata)
    else:
        _write_stl_binary(filename, stldata)

    print('wrote {} triangles to {}'.format(ntris, filename))

######################################################################
# make half-edge data structure to describe this mesh
#
# given ntris triangles, this returns a matrix of shape (ntris*3, 3)
# where each row describes a half-edge with entries
#
#   vidx0, vidx1, hidx_opposite
#
# where vidx0 and vidx1 are the indices in to the vertices array
# of the vertices for this edge and hidx_opposite is the index
# of the half edge (i.e. row index of returned matrix) whose
# vertices are ordered in the opposite direction (i.e. as vidx1, vidx0)
#
# to convert between half-edge indices and face indices:
#
#   face_index for hidx is (hidx // 3)
#   base_index for hidx is face_index * 3
#   next_index for hidx is base_index + (hidx + 1) % 3
#   hidx for face_index is face_index * 3
#
# call make_vertex_half_edge_lookup below if you need a mapping from
# vertex indices to half-edge indices.

def make_half_edges(triangles):

    ntris = len(triangles)
        
    pairs = [ [0, 1], [1, 2], [2, 0] ]

    half_edges = np.empty((ntris*3, 3), dtype=int)
    
    edge_pairs = half_edges[:, :2]

    edge_pairs[:] = triangles[:, pairs].reshape(-1, 2)
    half_edges[:, 2] = -1
    
    edge_lookup = dict()

    for hidx, key in enumerate(tuple(x) for x in edge_pairs):
        assert key not in edge_lookup
        edge_lookup[key] = hidx


    for key, hidx in edge_lookup.items():
        rkey = key[::-1]
        if rkey in edge_lookup:
            assert half_edges[hidx, 2] == -1
            half_edges[hidx, 2] = edge_lookup[rkey]

    return half_edges

######################################################################
# get a single half-edge corresponding to each vertex

def make_vertex_half_edge_lookup(vertices, half_edges):

    vidx0, hidx_for_vidx0 = np.unique(half_edges[:, 0], return_index=True)

    hidx_by_vertex = -np.ones(len(vertices), dtype=int)
    hidx_by_vertex[vidx0] = hidx_for_vidx0
            
    return half_edges, hidx_by_vertex
    
######################################################################
# get "canonical" half edges where either a) vidx0 < vidx1 or b) there
# is no opposing half-edge

def get_half_edge_index(half_edges):

    he_vidx0 = half_edges[:,0]
    he_vidx1 = half_edges[:,1]
    he_opp   = half_edges[:,2]
    
    he_idx, = np.nonzero( (he_vidx0 < he_vidx1) | (he_opp < 0) )

    return he_idx

######################################################################
# trim original polygon

def make_trimmed_polygon(contour, cutting_polygons):

    trimmed_polygon = sgeom.Polygon(contour)
    for cutting_polygon in cutting_polygons:
        trimmed_polygon = trimmed_polygon.difference(sgeom.Polygon(cutting_polygon))

    if not isinstance(trimmed_polygon, sgeom.Polygon):
        plot_polygon(trimmed_polygon)
        plt.axis('equal')
        plt.show()
        raise RuntimeError('result of cutout is not a simple polygon :(')

    trimmed_coords = get_coords_as_array(trimmed_polygon)
    
    trimmed_coords = cleanup_small_edges(trimmed_coords)
    validate_polygon(trimmed_coords)

    trimmed_polygon = sgeom.Polygon(trimmed_coords)

    return trimmed_coords, trimmed_polygon


######################################################################
# make a PDF for laser cutting

def dump_svg(filename, sx, sy, polygon):

    sx = int(np.ceil(sx))
    sy = int(np.ceil(sy))

    w = 2*sx
    h = 2*sy

    polygon = (polygon * [1, -1]) + (sx, sy)

    commands = []
    command = 'M'
    
    for x, y in polygon:
        commands.append('{} {:.4f} {:.4f}'.format(command, x, y))
        command = 'L'

    commands.append('Z')

    path_data = ' '.join(commands)

    with open(filename, 'w') as ostr:

        ostr.write('<?xml version="1.0" encoding="utf-8"?>\n')
        ostr.write('<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{}mm" height="{}mm" viewBox="0 0 {} {}">\n'.format(
            w, h, w, h))
        ostr.write('<path d="{}" style="fill:none;stroke:#000000;stroke-width:0.05mm" />\n'.format(path_data));
        ostr.write('</svg>\n')

    print('wrote', filename)

######################################################################
# make inset polygon by insetting trimmed polygon

def make_inset_polygon(trimmed_polygon, amount):

    inset_polygon = trimmed_polygon.buffer(-amount, resolution=8)

    if not isinstance(inset_polygon, sgeom.Polygon):
        raise RuntimeError('result of buffer is not a simple polygon :(')

    inset_coords = get_coords_as_array(inset_polygon)
    inset_coords = cleanup_small_edges(inset_coords)
    
    inset_polygon = sgeom.Polygon(inset_polygon)

    validate_polygon(inset_coords)

    return inset_coords, inset_polygon
    
######################################################################
# constrained delaunay triangulation of trimmed polygon + inset polygon

def triangulate(sx, sy,
                trimmed_coords, trimmed_polygon,
                inset_coords, inset_polygon,
                amount):

    pout, ppoly, pbuf = get_label_points(trimmed_polygon, inset_polygon, 0.5*amount)


    box_coords = np.array([[sx, sy], [-sx, sy], [-sx, -sy], [sx, -sy]])
    
    vertices = np.vstack((box_coords, trimmed_coords, inset_coords))
    
    segments = np.vstack((make_segments(4, 0),
                          make_segments(len(trimmed_coords), 4),
                          make_segments(len(inset_coords), 4+len(trimmed_coords))))

    assert segments.min() == 0 and segments.max() == len(vertices) - 1

    regions = np.array([
        [pout.x, pout.y, 0, 0],
        [ppoly.x, ppoly.y, 2, 0],
        [pbuf.x, pbuf.y, 1, 0]])

    A = dict(vertices=vertices, segments=segments, regions=regions)
    B = tr.triangulate(A, 'pA')

    vertices = B['vertices']
    triangles = B['triangles']
    attributes = B['triangle_attributes'].flatten().astype(np.int32)

    return vertices, triangles, attributes

######################################################################
# make 3D mesh from triangulation

def make_mesh(vertices, triangles, attributes, t, h):

    half_edges = make_half_edges(triangles)
    
    nbase = len(vertices)

    ones = np.ones((nbase, 1))

    v0 = np.hstack((vertices, 0*ones))
    v1 = np.hstack((vertices, t*ones))
    v2 = np.hstack((vertices, h*ones))

    v3d = np.vstack((v0, v1, v2))
    
    t3d = []

    for face_idx, triangle in enumerate(triangles):
        level = attributes[face_idx]
        if not level:
            continue
        t3d.append(triangle + nbase*level)
        t3d.append(triangle[::-1])
        for hidx in range(face_idx*3, face_idx*3+3):
            vidx0, vidx1, hidx_opposite = half_edges[hidx]
            if hidx_opposite >= 0:
                face_idx_opposite = hidx_opposite // 3
                level_opposite = attributes[face_idx_opposite]
                if level == 2 and level_opposite == 0:
                    t3d.append([vidx1 + nbase*2,
                                vidx0 + nbase*2,
                                vidx0])
                    t3d.append([vidx0, 
                                vidx1,
                                vidx1 + nbase*2])
                elif level == 2 and level_opposite == 1:
                    t3d.append([vidx1 + nbase*2,
                                vidx0 + nbase*2,
                                vidx0 + nbase])
                    t3d.append([vidx0 + nbase, 
                                vidx1 + nbase,
                                vidx1 + nbase*2])
                        
    t3d = np.array(t3d)

    h3d = make_half_edges(t3d)
    
    is_manifold = np.all(h3d[:,2] >= 0)
    assert is_manifold

    return v3d, t3d

######################################################################
# do the things
    
def main():

    opts = parse_cmdline()

    contour = get_path(opts.svgfile)

    sx, sy = contour.max(axis=0) + max(BBOX_MARGIN, opts.rim_width)

    validate_polygon(contour)

    cutter_profile = get_cutter_profile(opts)

    cutting_polygons = get_cutting_polygons(contour, cutter_profile)

    trimmed_coords, trimmed_polygon = make_trimmed_polygon(contour, 
                                                           cutting_polygons)

    svg_filename = 'foo.svg'
    stl_filename = 'foo.stl'

    dump_svg(svg_filename, sx, sy, trimmed_coords)

    inset_coords, inset_polygon = make_inset_polygon(trimmed_polygon,
                                                     opts.rim_width)

    vertices, triangles, attributes = triangulate(sx, sy,
                                                  trimmed_coords, trimmed_polygon,
                                                  inset_coords, inset_polygon,
                                                  opts.rim_width)

    v3d, t3d = make_mesh(vertices, triangles, attributes,
                         opts.base_thickness,
                         opts.rim_height)

    write_stl(stl_filename, v3d, t3d, 'binary')

if __name__ == '__main__':
    main()
