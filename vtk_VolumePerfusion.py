import sys
import os
import vtk
import numpy as np
import time
import glob
import math
import argparse
from collections import defaultdict
from tqdm import tqdm

def read_polydata(filename, datatype=None):
    """
    Load the given file, and return a vtkPolyData object for it.
    Args:
        filename (str): Path to input file.
        datatype (str): Additional parameter for vtkIdList objects.
    Returns:
        polyData (vtkSTL/vtkPolyData/vtkXMLStructured/
                    vtkXMLRectilinear/vtkXMLPolydata/vtkXMLUnstructured/
                    vtkXMLImage/Tecplot): Output data.
    """

    # Check if file exists
    if not path.exists(filename):
        raise RuntimeError("Could not find file: %s" % filename)

    # Check filename format
    fileType = filename.split(".")[-1]
    if fileType == '':
        raise RuntimeError('The file does not have an extension')

    # Get reader
    if fileType == 'stl':
        reader = vtk.vtkSTLReader()
        reader.MergingOn()
    elif fileType == 'vtk':
        reader = vtk.vtkPolyDataReader()
    elif fileType == 'vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif fileType == 'vts':
        reader = vtk.vtkXMinkorporereLStructuredGridReader()
    elif fileType == 'vtr':
        reader = vtk.vtkXMLRectilinearGridReader()
    elif fileType == 'vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fileType == "vti":
        reader = vtk.vtkXMLImageDataReader()
    elif fileType == "np" and datatype == "vtkIdList":
        result = np.load(filename).astype(np.int)
        id_list = vtk.vtkIdList()
        id_list.SetNumberOfIds(result.shape[0])
        for i in range(result.shape[0]):
            id_list.SetId(i, result[i])
        return id_list
    else:
        raise RuntimeError('Unknown file type %s' % fileType)

    # Read
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    return polydata

def write_polydata(input_data, filename, datatype=None):
    """
    Write the given input data based on the file name extension.
    Args:
        input_data (vtkSTL/vtkPolyData/vtkXMLStructured/
                    vtkXMLRectilinear/vtkXMLPolydata/vtkXMLUnstructured/
                    vtkXMLImage/Tecplot): Input data.
        filename (str): Save path location.
        datatype (str): Additional parameter for vtkIdList objects.
    """
    # Check filename format
    fileType = filename.split(".")[-1]
    if fileType == '':
        raise RuntimeError('The file does not have an extension')

    # Get writer
    if fileType == 'stl':
        writer = vtk.vtkSTLWriter()
    elif fileType == 'vtk':
        writer = vtk.vtkPolyDataWriter()
    elif fileType == 'vts':
        writer = vtk.vtkXMLStructuredGridWriter()
    elif fileType == 'vtr':
        writer = vtk.vtkXMLRectilinearGridWriter()
    elif fileType == 'vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif fileType == 'vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif fileType == "vti":
        writer = vtk.vtkXMLImageDataWriter()
    elif fileType == "np" and datatype == "vtkIdList":
        output_data = np.zeros(input_data.GetNumberOfIds())
        for i in range(input_data.GetNumberOfIds()):
            output_data[i] = input_data.GetId(i)
        output_data.dump(filename)
        return
    else:
        raise RuntimeError('Unknown file type %s' % fileType)

    # Set filename and input
    writer.SetFileName(filename)
    writer.SetInputData(input_data)
    writer.Update()

    # Write
    writer.Write()


def intializeVTP(filename,vprint):
	"""
	Read vtp file.
	"""
	datareader=vtk.vtkXMLPolyDataReader()
	datareader.SetFileName(filename)
	datareader.Update()

	mesh=vtk.vtkDataSetMapper()
	mesh=datareader.GetOutput()
	vprint('Loaded .vtp file.')
	return mesh

def intializeVTU(filename,vprint):
	"""
	Read vtu file.
	"""
	datareader=vtk.vtkXMLUnstructuredGridReader()
	datareader.SetFileName(filename)
	datareader.Update()

	mesh=datareader.GetOutput()
	vprint('Loaded .vtu file.')
	return mesh

def writeVTU(mesh,filename):
	"""
	write vtu file.
	"""
	print('Writing .vtu file...')
	w = vtk.vtkXMLUnstructuredGridWriter()
	w.SetInputData(mesh)
	w.SetFileName(filename)
	w.Write()
	print('done.')

def writeVTP(mesh,filename):
	"""
	Write vtp file.
	"""
	w = vtk.vtkXMLUnstructuredDataWriter()
	w.SetInputData(mesh)
	w.SetFileName(filename + '.vtp')
	w.Write()

def calcDistanceAlongSurface(mesh,startPt,endPt):
	"""
	Calculate the distance along mesh surface between 2 points in a given vtk model
	args:
		mesh (vtkObject) : vtk model that points referenced from
		startPt (int) : ID of starting point
		endPt (int) : ID of end point
	return:
		dist (float) : distance between points along surface
	"""
	dijkstra = vtk.vtkDijkstraGraphGeodesicPath()
	dijkstra.SetInputData(mesh)
	dijkstra.SetStartVertex(startPt)
	dijkstra.SetEndVertex(endPt)
	dijkstra.Update()
	pts = dijkstra.GetOutput().GetPoints()
	dist = 0.0
	if(pts is not None):
		for ptId in range(pts.GetNumberOfPoints()-1):
			pts.GetPoint(ptId, p0)
			pts.GetPoint(ptId+1, p1)
			dist += math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p0, p1))
	return dist

def calculateCapCenters(caps):
	"""
	Calculate the distance along mesh surface between 2 points in a given vtk model
	args:
		caps (list(vtkObject)) : list of cap vtk objects
	returns:
		cap_centers (list(tuple)) : list of cap center coordinates (x,y,z)
	"""
	cap_centers = []
	for cap in caps:
		numPts = cap.GetNumberOfPoints()
		center_x,center_y,center_z = 0,0,0
		for p in range(0,numPts):
			x,y,z = cap.GetPoint(p)
			center_x += x
			center_y += y
			center_z += z
		center_x = center_x/numPts
		center_y = center_y/numPts
		center_z = center_z/numPts
		cap_centers.append([center_x,center_y,center_z])
	return cap_centers

def minDistanceBetweenPointslist(model, seedPt, pt_list):
	"""
	Calculate the minimum distance between a given point and a list of points
	args:
		caps (list(vtkObject)) : list of cap vtk objects
	returns:
		min (float) : minimum distance
		min_pt_index (int) : index of the point in the list that is closest to seed pt
	"""
	min = calcDistance2Points(model, seedPt,pt_list[0])
	min_pt_index = 0
	for iPt in range(0,len(pt_list)):
		distance = calcDistance2Points(model, seedPt,pt_list[iPt])
		if(distance < min ):
			min = distance
			min_pt_index = iPt
	return min,min_pt_index

def convert_np_array_to_vtk(name,np_array):
	data_array = vtk.vtkDoubleArray()
	data_array.SetName(name)
	for i in range(0,len(np_array)):
		data_array.InsertNextValue(np_array[i])
	return data_array

def getConnectedVerticesNotIncludingSeed(model, seedPt):
	cell_list = vtk.vtkIdList()
	connectedPts_list = vtk.vtkIdList()
	model.GetPointCells(seedPt,cell_list)
	for j in range(0,cell_list.GetNumberOfIds()):
		pt_list = vtk.vtkIdList()
		pt_list = model.GetCell(cell_list.GetId(j)).GetPointIds()
		for k in range(0,pt_list.GetNumberOfIds()):
			if (pt_list.GetId(k) != seedPt):
				connectedPts_list.InsertUniqueId(pt_list.GetId(k))
	return connectedPts_list

def calcDistance2Points(model, pt1,pt2):
	"""
	Calculate the euclidian distance between 2 points in a given vtk model
	args:
		model (vtkObject) : vtk model that points referenced from
		pt1 (int) : first point ID or (x,y,z)
		pt2 (int) : second point ID or (x,y,z)
	return:
		distance (float) : distance between points
	"""
	if(type(pt1) is int):
		x1,y1,z1 = model.GetPoint(pt1)
	elif(type(pt1) is list):
		x1,y1,z1 = pt1[0],pt1[1],pt1[2]
	else:
		vprint(type(pt1))
	if(type(pt2) is int):
		x2,y2,z2 = model.GetPoint(pt2)
	else:
		x2,y2,z2 = pt2[0],pt2[1],pt2[2]
	distance = ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)**(.5)
	return distance

def maxDistanceBetweenPoints(model, seedPt, connectedPts_list):
	"""
	Calculate the max euclidian distance between seed point and vtklist of points in a given vtk model
	args:
		model (vtkObject) : vtk model that points referenced from
		seedPt (int) : seed point ID
		connectedPts_list (vtkIdList) : list of  point IDs
	return:
		max (float) : max distance between points
	"""
	max = 0
	for i in xrange(0,connectedPts_list.GetNumberOfIds()):
		distance = calcDistance2Points(model, seedPt,connectedPts_list.GetId(i))
		if(distance > max):
			max = distance
	return max

def minDistanceBetweenPoints(model, seedPt, connectedPts_list):
	"""
	Calculate the max euclidian distance between seed point and vtklist of points in a given vtk model
	args:
		model (vtkObject) : vtk model that points referenced from
		seedPt (int) : seed point ID
		connectedPts_list (vtkIdList) : list of  point IDs
	return:
		min (float) : min distance between points
		min_pt_index (int) : pt index in connectedPts_list
	"""
	min = calcDistance2Points(model, seedPt,connectedPts_list.GetId(0))
	min_pt_index = 0
	for iPt in range(0,connectedPts_list.GetNumberOfIds()):
		distance = calcDistance2Points(model, seedPt,connectedPts_list.GetId(iPt))
		if(distance < min):
			min = distance
			min_pt_index = iPt
	return min,min_pt_index

def minDistanceBetweenPointsinSet(model, seedPt, connectedPts_list):
	"""
	Calculate the max euclidian distance between seed point and vtklist of points in a given vtk model
	args:
		model (vtkObject) : vtk model that points referenced from
		seedPt (int) : seed point ID
		connectedPts_list (set) : set of  point IDs
	return:
		min (float) : min distance between points
		min_pt (int) : pt in connectedPts_list
	"""
	min = calcDistance2Points(model, seedPt,0)
	min_pt = list(connectedPts_list)[0]
	for iPt in connectedPts_list:
		distance = calcDistance2Points(model, seedPt,iPt)
		if(distance < min):
			min = distance
			min_pt = iPt
	return min,min_pt

def minDistanceBetweenPointsGraph(graph, heart, ip, cap_center_coordinates):
	"""
	Calculate the max euclidian distance between seed point and vtklist of points in a given vtk model
	args:
		model (vtkObject) : vtk model that points referenced from
		seedPt (int) : seed point ID
		connectedPts_list (set) : set of  point IDs
	return:
		min (float) : min distance between points
		min_pt (int) : pt in connectedPts_list
	"""
	destinations  = set()
	for i in cap_center_coordinates:
		destinations.add(heart.FindPoint(i))
	visited,path,point = dijsktra_closest(graph,ip,destinations)
	min, min_index = minDistanceBetweenPointslist(heart, point, pt_list)
	return visited, min_index

class Graph:
	"""
	Graph class
	"""
	def __init__(self):
		self.nodes = set()
		self.edges = defaultdict(list)
		self.distances = {}

	def add_node(self, value):
		self.nodes.add(value)

	def add_edge(self, from_node, to_node, distance):
		self.edges[from_node].append(to_node)
		self.edges[to_node].append(from_node)
		self.distances[(from_node, to_node)] = distance
		self.distances[(to_node, from_node)] = distance

	def add_virtual_node(self, v_node, zero_edge_nodes):
		self.nodes.add(v_node)
		for i in zero_edge_nodes:
			self.edges[v_node].append(i)
			self.edges[i].append(v_node)
			self.distances[(v_node, i)] = 0
			self.distances[(i, v_node)] = 0

	def add_virtual_node_distances(self, v_node, edge_nodes, distances):
		self.nodes.add(v_node)
		DISTANCE_CUTOFF = 1
		for i in edge_nodes:
			if(distances[i]<DISTANCE_CUTOFF):
				self.edges[v_node].append(i)
				self.edges[i].append(v_node)
				self.distances[(v_node, i)] = distances[i]
				self.distances[(i, v_node)] = distances[i]

	def get_num_of_nodes(self):
		return len(self.nodes)

def dijsktra(graph, initial):
	"""
	Dijsktra algorithm that vists all nodes in graph from initial node.
	args:
		graph (Graph) : vtk model that points referenced from
		initial (int) : seed point ID
	return:
		visited (dict) : distance from inital point to key point
		path (dict) : directed path [from_node] = to_node
	"""
	visited = {}
	visited[initial] = 0
	path = {}
	path_nodes = set()

	nodes = set(graph.nodes)
	counter = 0
	pbar = tqdm(total=len(nodes))
	while nodes: 
		pbar.update(1)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] <= visited[min_node]:
					min_node = node
		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in graph.edges[min_node]:
			weight = current_weight + graph.distances[(min_node, edge)]
			if edge not in visited or weight <= visited[edge]:
				visited[edge] = weight
				path[edge] = min_node
				path_nodes.add(edge)
		counter += 1
	pbar.close()
	return visited, path



def dijsktra_closest(graph, initial, destinations):
	"""
	Dijsktra algorithm that vists all nodes in graph from initial node until all destinations are reached.
	args:
		graph (Graph) : vtk model that points referenced from
		initial (int) : seed point ID
		destinations (set) : set of destination IDs
	return:
		visited (dict) : distance from inital point to key point
		path (dict) : directed path [from_node] = to_node
	"""
	visited = {initial: 0}
	path = {}
	path_nodes = set()
	nodes = set(graph.nodes)
	if initial in destinations:
		destinations.remove(initial)

	while len(destinations.intersection(path_nodes))==0: 
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] < visited[min_node]:
					min_node = node
		print(min_node)
		if min_node is None:
			break

		nodes.remove(min_node)
		current_weight = visited[min_node]

		for edge in graph.edges[min_node]:
			weight = current_weight + graph.distances[(min_node, edge)]
			if edge not in visited or weight < visited[edge]:
				visited[edge] = weight
				path[edge] = min_node
				path_nodes.add(min_node)

	return visited, path, destinations.intersection(path)

def weightedDijsktra(graph, initial, weights):
	"""
	Dijsktra algorithm that vists all nodes in graph from initial node weighted by weights.
	args:
		graph (Graph) : vtk model that points referenced from
		initial (int) : seed point ID
		weights (dict) : weights for every initial seed point
	return:
		visited (dict) : distance from inital point to key point
		path (dict) : directed path [from_node] = to_node
	"""
	visited = {}
	visited[initial] = 0
	path = {}
	path_nodes = set()

	nodes = set(graph.nodes)
	counter = 0
	vprint('Starting to build distance map...')
	pbar = tqdm(total=len(nodes))
	while nodes: 
		pbar.update(1)
		min_node = None
		for node in nodes:
			if node in visited:
				if min_node is None:
					min_node = node
				elif visited[node] <= visited[min_node]:
					min_node = node
		if min_node is None:
			break

		nodes.remove(min_node)
		current_dist = visited[min_node]

		for edge in graph.edges[min_node]:
			if min_node==initial:
				dist = current_dist + graph.distances[(min_node, edge)]
			else:
				dist = current_dist + graph.distances[(min_node, edge)]*weights[min_node]
			if edge not in visited or dist <= visited[edge]:
				visited[edge] = dist
				path[edge] = min_node
				path_nodes.add(edge)
				if min_node in weights:
					weights[edge] = weights[min_node]
		counter += 1
	pbar.close()
	return visited, path	

def shortest_path(graph, origin, destination):
	"""
	Determines shortest path from origin to destination on graph.
	"""
	visited, paths = dijkstra(graph, origin)
	full_path = deque()
	_destination = paths[destination]

	while _destination != origin:
		full_path.appendleft(_destination)
		_destination = paths[_destination]

	full_path.appendleft(origin)
	full_path.append(destination)

	return visited[destination], list(full_path)

def fastMarching(heart_graph,heart,seedPts):
	"""
	Calculate the distance of every point to seed pts using fast marching algorithm
	"""
	pt_set= set()
	numPts = heart.GetNumberOfPoints()
	for ptID in seedPts:
		pt_set.add(heart.FindPoint(ptID))

	#intialize edge list
	edgePt = set()
	temp_list = vtk.vtkIdList()
	pt_dist = {}
	for pt in  pt_set:
		connnectedPt_list = getConnectedVerticesNotIncludingSeed(heart,pt)
		for j in range(0,connnectedPt_list.GetNumberOfIds()):
			# new point to decide whether to add to patch, edge, or nothing (if already in edge)
			cpt = connnectedPt_list.GetId(j)
			pt_dist[cpt] = calcDistance2Points(heart,pt,cpt)
			temp_list.InsertNextId(cpt)
	for i in range(0,temp_list.GetNumberOfIds()):
		edgePt.add(temp_list.GetId(i))
		pt_set.add(temp_list.GetId(i))

	temp_list = vtk.vtkIdList()
	#search until all points are found
	while(len(edgePt) > 0):
		temp = set()
		for i in  edgePt:
			connnectedPt_list = getConnectedVerticesNotIncludingSeed(heart,i)
			for j in range(0,connnectedPt_list.GetNumberOfIds()):
				# new point to decide whether to add to patch, edge, or nothing (if already in edge)
				cpt = connnectedPt_list.GetId(j)
				if(cpt in pt_set and cpt in pt_dist):
					pt_set.add(i)
					pt_dist[i] = pt_dist[cpt] + calcDistance2Points(heart,i,cpt)
					heart_graph.add_edge(i,cpt,calcDistance2Points(heart,i,cpt))
				elif(connnectedPt_list.GetId(j) not in pt_set and cpt not in edgePt):
					temp.add(cpt)
		edgePt = temp
	data_array = np.zeros(numPts)
	for i in pt_dist:
		data_array[i] = pt_dist[i]

	vtk_array = vtk.vtkDoubleArray()
	for i in data_array:
		vtk_array.InsertNextValue(i)
	vtk_array.SetName('Point Distances')
	heart.GetPointData().AddArray(vtk_array)

	return pt_dist

def multipleSourceDistance(heart,graph,v_node,child_nodes,distances,weights):
	"""
	calculates the distance of every point on heart mesh to nearest child node.
	"""
	graph.add_virtual_node_distances(v_node,child_nodes,distances)
	visited,path = weightedDijsktra(graph,v_node,weights)
	data_array = vtk.vtkDoubleArray()
	data_array.SetName('distance_map')
	for i in range(0,heart.GetNumberOfPoints()):
		if i in visited:
			data_array.InsertNextValue(visited[i])
		else:
			data_array.InsertNextValue(-1)
	heart.GetPointData().AddArray(data_array)
	return heart

def multipleCapSourceDistance(heart,graph,v_node,child_nodes):
	"""
	calculates the distance of every point on heart mesh to nearest cap.
	"""
	graph.add_virtual_node_distances(v_node,child_nodes)
	visited,path = weightedDijsktra(graph,v_node,weights)
	data_array = vtk.vtkDoubleArray()
	data_array.SetName('cap_distance_map')
	for i in range(0,heart.GetNumberOfPoints()):
		if i in visited:
			data_array.InsertNextValue(visited[i])
		else:
			data_array.InsertNextValue(-1)
	heart.GetPointData().AddArray(data_array)
	return heart

def generateGraph(heart):
	"""
	Generates a graph class representation of the heart mesh
	"""
	vprint('Generating graph...')
	heart_graph = Graph()
	print(heart.GetNumberOfPoints())
	for i in tqdm(range(0,heart.GetNumberOfPoints())):
		heart_graph.add_node(i)
		connnectedPt_list = getConnectedVerticesNotIncludingSeed(heart,i)
		for j in range(0,connnectedPt_list.GetNumberOfIds()):
			# new point to decide whether to add to patch, edge, or nothing (if already in edge)
			cpt = connnectedPt_list.GetId(j)
			heart_graph.add_edge(i,cpt,calcDistance2Points(heart,i,cpt))
	return heart_graph

def determinePerfusionVolumesMask(image,heart,threshold):
	"""
	Label the perfusion territory volumes
	"""
	numPts = heart.GetNumberOfPoints()
	heart_data = [0]*numPts
	print('Assigning Perfusion Volumes...\n')
	for ip in tqdm(range(0,numPts)):
		value = image.GetPointData().GetArray(0).GetValue(image.FindPoint(heart.GetPoint(ip)))
		if(value>threshold):
			heart_data[ip] = value
		else:
			heart_data[ip] = -1

	#generate summary data array for perfusion
	data_array = vtk.vtkDoubleArray()
	data_array.SetName('PerfusionVolumes')
	for ptID in range(0,numPts):
		data_array.InsertNextValue(heart_data[ptID])
	heart.GetPointData().AddArray(data_array)
	writeVTU(heart,'heart_mask.vtu')

def determineCapPerfusionVolumes(caps,cap_center_coordinates,heart):
	"""
	Calculate the perfusion territories based on all cap (outlet) locations
	"""
	numPts = heart.GetNumberOfPoints()
	heart_data = [0]*numPts
	heart_graph = Graph()

	cap_heart_points = set()
	for i in cap_center_coordinates:
		cap_heart_points.add(heart.FindPoint(i))
	heart_graph = generateGraph(heart)
	heart = multipleCapSourceDistance(heart,heart_graph,100000000000,cap_heart_points)

	for i in range(0,numPts):
		connnectedPt_list = getConnectedVerticesNotIncludingSeed(heart,i)
		for j in range(0,connnectedPt_list.GetNumberOfIds()):
			# new point to decide whether to add to patch, edge, or nothing (if already in edge)
			cpt = connnectedPt_list.GetId(j)
			heart_graph.add_edge(i,cpt,calcDistance2Points(heart,i,cpt))
			heart_graph.add_edge(cpt,i,calcDistance2Points(heart,i,cpt))
	print(cap_heart_points)
	for i in range(0,len(cap_center_coordinates)):
		if heart.FindPoint(cap_center_coordinates[i]) in heart_graph.nodes:
			visited, path = dijsktra(heart_graph,heart.FindPoint(cap_center_coordinates[i]))
			data_array = vtk.vtkDoubleArray()
			data_array.SetName(caps[i] + '_distance_map')
			for i in range(0,numPts):
				if i in visited:
					data_array.InsertNextValue(visited[i])
				else:
					data_array.InsertNextValue(-1)
			heart.GetPointData().AddArray(data_array)
			writeVTU(heart,'heart_distance_mapped.vtu')

	for ip in range(0,numPts):
		value = image.GetPointData().GetArray(0).GetValue(image.FindPoint(heart.GetPoint(ip)))
		min,min_pt_index = minDistanceBetweenPointsGraph(heart_graph, heart, ip, cap_center_coordinates)
		heart_data[ip] = min_pt_index
	#generate summary data array for perfusion
	data_array = vtk.vtkDoubleArray()
	data_array.SetName('PerfusionVolumes')
	for ptID in range(0,numPts):
		data_array.InsertNextValue(heart_data[ptID])
	heart.GetPointData().AddArray(data_array)

	perfusion_data = np.zeros((len(caps),numPts))
	for ip in range(0,len(heart_data)):
		if(heart_data[ip]>=0):
			perfusion_data[heart_data[ip],ip] = 1

	#generate separate data array for each perfusion volume
	#calculate the volume of each perfused area of each cap
	volumes = []
	LCA_data = np.zeros(numPts)
	RCA_data = np.zeros(numPts)
	RSA_data = np.zeros(numPts)
	CA_data = np.zeros(numPts)
	cap_pt_list = vtk.vtkIdList()
	for i in tqdm(range(0,len(perfusion_data))):
		data = vtk.vtkDoubleArray()
		data.SetName(str(i) + '_' + caps[i])
		for ip in range(0,numPts):
			if(perfusion_data[i,ip]>0):
				cap_pt_list.InsertNextId(ip)
				if(caps[i].startswith('LCA')):
					LCA_data[ip] = perfusion_data[i,ip]
					CA_data[ip] = 1
				elif(caps[i].startswith('RCA')):
					RCA_data[ip] = perfusion_data[i,ip]
					CA_data[ip] = 2
				elif(caps[i].startswith('RSA')):
					RSA_data[ip] = perfusion_data[i,ip]
					CA_data[ip] = 3
			data.InsertNextValue(perfusion_data[i,ip])
		heart.GetPointData().AddArray(data)
		Mass = extractRegionVolume(heart,cap_pt_list)
		p2c = vtk.vtkPointDataToCellData()
		p2c.SetInputData(heart)
		p2c.PassPointDataOn()
		warp = vtk.vtkWarpVector()
		warp.SetInputConnection(p2c.GetOutputPort())
		thresh = vtk.vtkThreshold()
		thresh.SetInputConnection(warp.GetOutputPort())
		thresh.ThresholdBetween(i,i)
		thresh.SetInputArrayToProcess(1, 0, 0, 0, "PerfusionVolumes")
		volumes.append(Mass.GetVolume())
		cap_pt_list.Reset()

	heart.GetPointData().AddArray(convert_np_array_to_vtk('LCA_all',LCA_data))
	heart.GetPointData().AddArray(convert_np_array_to_vtk('RCA_all',RCA_data))
	heart.GetPointData().AddArray(convert_np_array_to_vtk('RSA_all',RSA_data))
	heart.GetPointData().AddArray(convert_np_array_to_vtk('CA_all',CA_data))

	
	return volumes

def determineCenterlinePerfusionVolumes(coordinates,weights,heart,out_filename):
	"""
	Calculate the perfusion territories based on centerline points
	"""
	numPts = heart.GetNumberOfPoints()
	heart_data = [0]*numPts
	heart_graph = Graph()

	cap_heart_points = set()
	weight_dict = {}
	distances = {}

	for i in range(0,len(coordinates)):
		
		cap_heart_points.add(heart.FindPoint(coordinates[i]))
		
		weight_dict[heart.FindPoint(coordinates[i])] = weights[i]
		distances[heart.FindPoint(coordinates[i])] = calcDistance2Points(heart,heart.FindPoint(coordinates[i]),coordinates[i])
		
	heart_graph = generateGraph(heart)

	fastMarching(heart_graph,heart,coordinates)
	heart = multipleSourceDistance(heart,heart_graph,-1,cap_heart_points,distances,weight_dict)
	writeVTU(heart,out_filename)

def extractRegionVolume(mesh,selection_nodes):
	#Intialize variables
	ids = vtk.vtkIdTypeArray()
	cell_nodes = vtk.vtkIdList()
	cell_vtk_Id_list = vtk.vtkIdList()
	cellIds = vtk.vtkIdTypeArray()
	ids.SetNumberOfComponents(1)
	
	#Determines the cells enclosed by selection_nodes (which are points)
	vprint('Number of nodes in this volume: ', selection_nodes.GetNumberOfIds())
	for i in range(0,selection_nodes.GetNumberOfIds()):
		ids.InsertNextValue(selection_nodes.GetId(i))
		mesh.GetPointCells(selection_nodes.GetId(i), cell_nodes)
		for j in range(0,cell_nodes.GetNumberOfIds()):
			cell_vtk_Id_list.InsertUniqueId(cell_nodes.GetId(j))

	#Converts the vtkIdList into vtkIdTypeArray
	for i in range(0,cell_vtk_Id_list.GetNumberOfIds()):
		cellIds.InsertNextValue(cell_vtk_Id_list.GetId(i))
	vprint('Number of cells in this volume: ', cell_vtk_Id_list.GetNumberOfIds())
	#Creates the selection object to extract the subset of cells from the mesh
	region=vtk.vtkExtractSelection()
	region.SetInputData(0,mesh)
	tempCells = vtk.vtkSelectionNode()
	tempCells.SetFieldType(vtk.vtkSelectionNode.CELL)
	tempCells.SetContentType(vtk.vtkSelectionNode.INDICES)
	tempCells.SetSelectionList(cellIds)
	tempSelection = vtk.vtkSelection()
	tempSelection.AddNode(tempCells)
	region.SetInputData(1,tempSelection)
	region.Update()

	#Outputs the mesh as an Mass object
	output = vtk.vtkPolyData()
	output.ShallowCopy(region.GetOutput())
	vprint(region.GetOutput().GetNumberOfCells())
	dssf = vtk.vtkDataSetSurfaceFilter()
	dssf.SetInputConnection(region.GetOutputPort())
	dssf.Update()
	Mass = vtk.vtkMassProperties()
	Mass.SetInputData(dssf.GetOutput())
	Mass.Update()
	return Mass

def writeVolumesToFile(filename,cap_names,volumes):
	outfile = open(filename,'w')
	out_string = 'Cap,Volume_Perfusion' + '\n'
	outfile.write(out_string)
	for i in range(0,len(volumes)):
		out_string = cap_names[i] + ',' + str(volumes[i]) + '\n'
		outfile.write(out_string)
	outfile.close()

def writeFlowFile(filename,flow):
	outfile = open(filename,'w')
	out_string = '# Time (sec)\tFlow (micrometers^3/sec)' + '\n'
	outfile.write(out_string)
	out_string = '0.000000000' + '\t' + str(flow) + '\n'
	outfile.write(out_string)
	out_string = '1.000000000' + '\t' + str(flow) + '\n'
	outfile.write(out_string)
	outfile.close()

def getCoords(centerline):
	coords = []
	for i in range(0,centerline.GetNumberOfPoints()):
		coords.append(centerline.GetPoint(i))
	return coords

def getWeights(centerline,option):
	"""
	Sets weights based on the radius of the centerline point
	"""
	weights = []
	if option == 0:
		weights = [1]*centerline.GetNumberOfPoints()
	elif option==1:
		for i in range(0,centerline.GetNumberOfPoints()):
			weights.append(1/(centerline.GetPointData().GetArray('MaximumInscribedSphereRadius').GetValue(i))**2)
	return weights

def getNormWeights(centerline):
	"""
	Sets weights based on the radius of the centerline point and normalizes it to max radius
	"""
	weights = []
	sum = 0
	for i in range(0,centerline.GetNumberOfPoints()):
		sum += centerline.GetPointData().GetArray('MaximumInscribedSphereRadius').GetValue(i)
	norm_weight = sum/centerline.GetNumberOfPoints()
	for i in range(0,centerline.GetNumberOfPoints()):
		weights.append(norm_weight/(centerline.GetPointData().GetArray('MaximumInscribedSphereRadius').GetValue(i)))
	return weights

def markTerritories(heart,vtk_centerline_data,centerlines):
	"""
	Labels the perfusion territory based on closest centerline
	"""
	vtk_data = vtk.vtkDoubleArray()
	for pt in range(0,heart.GetNumberOfPoints()):
		min_centerline = list(centerlines.keys())[0]
		min_centerline_value = vtk_centerline_data[min_centerline].GetValue(pt)
		for i in vtk_centerline_data:
			if vtk_centerline_data[i].GetValue(pt) < min_centerline_value:
				min_centerline = i
				min_centerline_value = vtk_centerline_data[i].GetValue(pt)
		vtk_data.InsertNextValue(centerlines[min_centerline])
	vtk_data.SetName('Centerline_dist_map')
	heart.GetPointData().AddArray(vtk_data)
	writeVTU(heart,'heart_all_centerline.vtu')

def getCenterline(centerline_main,i):
	return

def update_progress(progress, total, vprint):  
	vprint('\r[{0:10}]{1:>2}'.format('#' * int(progress * 10 /total), progress))

def createParser():
	parser = argparse.ArgumentParser(description='Finds volume of tissue perfused by each outlet.')
	parser.add_argument('caps', type=str, help='the input model cap locations')
	parser.add_argument('image_data', type=str, help='the image filename (include file ext)')
	parser.add_argument('heart', type=str, help='the heart filename (include file ext)')
	parser.add_argument('data', type=str, help='the output filename (include file ext)')
	parser.add_argument('vtu_data', type=str, help='the output vtu filename (include file ext)')
	parser.add_argument('centerline', type=str, help='the input centerline (include file ext)')
	parser.add_argument('-t', '-threshold', type=float, nargs='?', default=1, help='threshold of heart tissue')
	parser.add_argument('-v', '-verbose', type=int, nargs='?', const=1, default=0, help='turn on verbosity')
	return parser

def main(args):
	IMAGE_FILENAME = args.image_data
	THRESHOLD = args.t
	DATA_FILENAME = args.data
	if not os.path.exists('./TEST_DATA'):
		os.mkdir('TEST_DATA')

	#Read vti file
	ref = vtk.vtkXMLImageDataReader()
	ref.SetFileName(IMAGE_FILENAME)
	ref.Update()
	#Read your data into another polydata variable for reading
	image=vtk.vtkPolyData()
	image=ref.GetOutput()
	global vprint
	if args.v:
	    def vprint(*args):
	      # Print each argument separately so caller doesn't need to
	      # stuff everything to be printed into a single string
	      for arg in args:
	        print(arg),
	else:
		vprint = lambda *a: None
	heart = intializeVTU(args.heart,vprint)
	#determinePerfusionVolumesMask(image,heart,args.t)

	# vtp files of the caps to calculate
	caps = []
	cap_names = []
	for file in os.listdir(args.caps):
		if file.endswith('.vtp') and not file.startswith('wall') and not file.startswith('aorta'):
			cap_names.append(file[:len(file)-4])
			cap = intializeVTP(args.caps + file,vprint)
			caps.append(cap)

	vprint('Found ' + str(len(caps)) + ' caps.')

	vtk_centerline_data = dict()
	for i in centerlines:
		if not os.path.isfile('./'+'heart_' + i.split('.')[0] + '.vtu'):

			cap_center_coordinates = calculateCapCenters(caps)
			
			
			
			cap_volumes = determineCapPerfusionVolumes(cap_names,cap_center_coordinates,heart)
		centerline_heart = intializeVTU('heart_' + i.split('.')[0] + '.vtu',vprint)
		vtk_centerline_data[i.split('.')[0]] = centerline_heart.GetPointData().GetArray('distance_map')
	if len(vtk_centerline_data) == len(centerlines):
		markTerritories(heart,vtk_centerline_data,centerline_dict)


	writeVTU(heart,args.vtu_data)

if __name__ == '__main__':
	parser = createParser()
	args = parser.parse_args()
	main(args)