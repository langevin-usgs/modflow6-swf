import numpy as np
import shapely
import flopy


def global_reach_number(ia_seg2reach, i, ilr):
    return ia_seg2reach[i] + ilr

CELLID_DICT = {
    "dis": [("layer", int), ("row", int), ("column", int)],
    "disv": [("layer", int), ("icell", int)],
    "disu": [("node", int)],
}


def get_downstream_segments(upstream_segments):
    ds = [[] for i in range(len(upstream_segments))]
    for iu, segs in enumerate(upstream_segments):
        for id in segs:
            if iu not in ds[id]:
                ds[id].append(iu)
    return ds    


class StreamUtil():

    def __init__(self, stream_segments, upstream_segments, modelgrid=None):
        self.stream_segments = stream_segments
        self.upstream_segments = upstream_segments
        self.downstream_segments = get_downstream_segments(upstream_segments)
        self.modelgrid = modelgrid
        self.gridtype = None
        if modelgrid is not None:
            gridtype = {"structured": "dis", "vertex": "disv", "unstructured": "disu"}
            self.gridtype = gridtype[modelgrid.grid_type]
        self._process()
        return
    
    def get_sfr_reachdata(self):
        sfr_reach_data = {
            "cellid": self.sfr_reach_cellid, 
            "length": self.sfr_reach_length, 
            "linestring": self.sfr_reach_linestring
        }
        return sfr_reach_data
    
    def get_sfr_packagedata(self):  #, layer, rwid, slope, rbth, rhk, roughness, ustrf, ndv):
        package_data = np.zeros(
            self.nreaches, 
            dtype=self.packagedata_dtype(self.gridtype)
        )
        package_data = package_data.view(np.recarray)
        # package_data = flopy.mf6.ModflowGwfsfr.packagedata.empty(gwf, maxbound=self.nreaches, cellid_expanded=True)

        # feature number
        package_data["ifno"] = np.arange(self.nreaches)

        # cell id
        if self.gridtype == "dis":
            row, col = zip(*self.sfr_reach_cellid)
            package_data["row"] = row
            package_data["column"] = col
            cellid_idx = (np.array(row), np.array(col))
        elif self.gridtype == "disv":
            package_data["icell"] = self.sfr_reach_cellid
        elif self.gridtype == "disu":
            package_data["node"] = self.sfr_reach_cellid

        # reach length
        package_data["rlen"] = self.sfr_reach_length

        # rtp (elevation of the stream bed bottom)
        package_data["rtp"] = self.modelgrid.top[cellid_idx]

        # ncon (number of connections for each reach)
        package_data["ncon"] = [
            len(self.reach_connectivity[i]) - 1 for i in range(self.nreaches)
        ]

        return package_data
    
    def _process(self):
        self._reinitialize()
        if self.modelgrid is not None:
            self._intersect()
        self._calculate_ia()
        self._build_network()
        return

    def _reinitialize(self):
        self.nsegments = len(self.stream_segments)
        self.reach_connectivity = []
        self.sfr_reach_cellid = []
        self.sfr_reach_length = []
        self.sfr_reach_linestring = []
        self.ia_seg2reach = None
        self.nreaches = None
        return

    def _intersect(self):
        # store intersection results
        self.intersection_results = []
        ixs = flopy.utils.GridIntersect(self.modelgrid)
        for stream_segment in self.stream_segments:
            v = ixs.intersect(shapely.LineString(stream_segment))
            self.intersection_results.append(v)
        return
        
    def _calculate_ia(self):
        # Sum the number of reaches for each segment
        ia_seg2reach = np.zeros(self.nsegments + 1, dtype=int)
        nrt = 0
        for iseg in range(self.nsegments):
            if self.modelgrid is None:
                # number of vertices in the segment minus 1
                nr = len(self.stream_segments[iseg]) - 1
            else:
                # number of cells
                v = self.intersection_results[iseg]
                nr = v.shape[0]
            ia_seg2reach[iseg + 1] = ia_seg2reach[iseg] + nr
            nrt += nr
        self.nreaches = ia_seg2reach[-1]
        self.ia_seg2reach = ia_seg2reach
        return

    def _build_network(self):

        # go through each segment and then each reach and number
        # the reaches from top to bottom using a global reach number
        for iseg in range(self.nsegments):

            # get intersection results
            #v = self.intersection_results[iseg]

            nreach_local = self.ia_seg2reach[iseg + 1] - self.ia_seg2reach[iseg]
            for ilocal_reach in range(nreach_local):
            #for ilocal_reach, reach in enumerate(v):

                # set indicators
                is_first_reach = ilocal_reach == 0
                #is_last_reach = ilocal_reach == v.shape[0] - 1
                is_last_reach = ilocal_reach == nreach_local - 1

                # start by adding reach number to reach connectivity
                igr = global_reach_number(self.ia_seg2reach, iseg, ilocal_reach)
                self.reach_connectivity.append([igr])

                if is_first_reach:
                    # add upstream reach from a different segment
                    for iup_seg in self.upstream_segments[iseg]:
                        igr_up = self.ia_seg2reach[iup_seg + 1] - 1
                        self.reach_connectivity[igr].append(igr_up)

                else:
                    # add global upstream reach in this same segment
                    igr_up = igr - 1
                    self.reach_connectivity[igr].append(igr_up)

                if is_last_reach:
                    # add last reach in this segment to the first reach
                    # in the next segment
                    for idn_seg in self.downstream_segments[iseg]:
                        igr_dn = self.ia_seg2reach[idn_seg]
                        self.reach_connectivity[igr].append(-igr_dn)

                else:
                    # add global downstream reach in this same segment
                    igr_dn = igr + 1
                    self.reach_connectivity[igr].append(-igr_dn)

                # store cellids, lengths, and shapes for subsequent processing
                if self.modelgrid is not None:
                    v = self.intersection_results[iseg]
                    reach = v[ilocal_reach]
                    self.sfr_reach_cellid.append(reach["cellids"])
                    self.sfr_reach_length.append(reach["lengths"])
                    self.sfr_reach_linestring.append(reach["ixshapes"])

        return

    def packagedata_dtype(self, gridtype):
        dtype = np.dtype(
            [
                ("ifno", int),
                *CELLID_DICT[gridtype],
                ("rlen", float),
                ("rwid", float),
                ("rgrd", float),
                ("rtp", float),
                ("rbth", float),
                ("rhk", float),
                ("man", float),
                ("ncon", int),
                ("ustrf", float),
                ("ndv", int),
            ]
        )
        return dtype
    
    def get_verts_iverts_by_linevertices(self):
        verts = []
        iverts = []
        for i in range(self.nreaches):
            iverts.append([])
        icell = 0
        for iseg in range(self.nsegments):
            line_segment = self.stream_segments[iseg]
            nreach_local = self.ia_seg2reach[iseg + 1] - self.ia_seg2reach[iseg]
            for ireach_local in range(nreach_local):
                v0 = line_segment[ireach_local]
                v1 = line_segment[ireach_local + 1]
                for xypoint in [v0, v1]:
                    nvert = len(verts)
                    if xypoint in verts:
                        iv = verts.index(xypoint)
                    else:
                        verts.append(xypoint)
                        iv = nvert
                    iverts[icell].append(iv)
                icell += 1
        return verts, iverts
    
    def get_verts_iverts_bygrid(self):
        ixs = self.intersection_results
        verts = []
        iverts = []
        for i in range(self.nreaches):
            iverts.append([])
        icell = 0
        for ix in ixs:
            segment_line = ix["vertices"]
            for reach_line in segment_line:
                for xypoint in reach_line:
                    nvert = len(verts)
                    if xypoint in verts:
                        iv = verts.index(xypoint)
                    else:
                        verts.append(xypoint)
                        iv = nvert
                    iverts[icell].append(iv)
                icell += 1
        return verts, iverts

    def get_vertices_cell2d(self, grid_based=False):
        if grid_based:
            if self.modelgrid is None:
                raise Exception("Cannot discretize reaches by model grid.  No model grid found.")
            verts, iverts = self.get_verts_iverts_bygrid()
        else:
            verts, iverts = self.get_verts_iverts_by_linevertices()
        vertices = [(iv, x, y) for iv, (x, y) in enumerate(verts)]
        cell2d = [(icell, 0.5, len(iverts[icell])) + tuple(iverts[icell]) for icell in range(self.nreaches)]
        return vertices, cell2d


def densify_geometry(line, step, keep_internal_nodes=True):
    xy = []  # list of tuple of coordinates
    lines_strings = []
    if keep_internal_nodes:
        for idx in range(1, len(line)):
            lines_strings.append(shapely.geometry.LineString(line[idx-1:idx+1]))
    else:
        lines_strings = [shapely.geometry.LineString(line)]
    
    for line_string in lines_strings:
        length_m = line_string.length # get the length
        for distance in np.arange(0, length_m + step, step):
            point = line_string.interpolate(distance)
            xy_tuple = (point.x, point.y)
            if xy_tuple not in xy:
                xy.append(xy_tuple)
        # make sure the end point is in xy
        if keep_internal_nodes:
            xy_tuple = line_string.coords[-1]
            if xy_tuple not in xy:
                xy.append(xy_tuple)
                   
    return xy