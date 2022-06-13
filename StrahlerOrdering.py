import pickle
import sys
from os import path
import vtk
import numpy as np
import time
import glob
import math
import argparse
from collections import defaultdict
from tqdm import tqdm
import copy


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

def intializeVTP(filename):
  datareader=vtk.vtkXMLPolyDataReader()
  datareader.SetFileName(filename)
  datareader.Update()

  mesh=vtk.vtkDataSetMapper()
  mesh=datareader.GetOutput()
  print('Loaded .vtp file.')
  return mesh

def intializeVTU(filename):
	datareader=vtk.vtkXMLUnstructuredGridReader()
	datareader.SetFileName(filename)
	datareader.Update()

	mesh=datareader.GetOutput()
	print('Loaded .vtu file.')
	return mesh

def writeVTU(mesh,filename):
	print('Writing .vtu file...')
	w = vtk.vtkXMLUnstructuredGridWriter()
	w.SetInputData(mesh)
	w.SetFileName(filename)
	w.Write()
	print('done.')

def writeVTP(mesh,filename):
  w = vtk.vtkXMLUnstructuredDataWriter()
  w.SetInputData(mesh)
  w.SetFileName(filename + '.vtp')
  w.Write()
# (C) Classify each segment into an initial order using the Strahler Ordering System
#	Terminal branches = order 1
#	When two branches of order n converge to 1 branch, converged branch = order n+1
#	If a higher order branch converges with a lower order branch, converged branch = higher order
def initStrahler(segDiam, segOrder, maxOrder, inletSegName, cl):
	# Define terminal branches as the end segment with no more joints (attached segment) as order 1
	# print("\n\nBefore Strahler Ordering")
	print(inletSegName)
	for seg in segName:
		if cl:
			if len(jointSeg[seg])<1:
				segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][0]) + " " + str(segOrder[int(segName[seg][0])]) )
		else:
			segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][-1]) + " " + str(segOrder[int(segName[seg][-1])]) )
			

	parentSeg = segName[inletSegName] #min(set(jointSeg)) # Str of Model's inlet segment
	maxGen = 0
	[segOrder, maxGen] = findStrahOrders(segOrder, parentSeg, maxGen)


	print("MAX GENERATION = " + str(maxGen))

	# print("\n\nAfter Strahler Ordering")
	for seg in segName:
		if cl:
			if not jointSeg[seg]:
				segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][0]) + " " + str(segOrder[int(segName[seg][0])]) )
		else:
			segOrder[seg] = 1
			# print(str(seg) + " " + str(segName[seg][-1]) + " " + str(segOrder[int(segName[seg][-1])]) )
			

	# print("\n\nJOINT SEG")
	# for vessel in segName:
	# 	for seg in segName[vessel]:
	# 		print(str(vessel) + " " + str(seg) + ": " + str(jointSeg[seg]))

	maxinitOrder = max(segOrder.values())
	orderTransl = maxOrder-maxinitOrder # Assumes MPA included
	for seg in segOrder:
		segOrder[seg] = segOrder[seg] + orderTransl


	return(segOrder, maxGen)

def calcPhyResistance(lengths,diameters):
	res = dict()
	for i in lengths:
		res[i] = 8*.00125*lengths[i]/(3.1415*(diameters[i]/2)**4)
	return res

def getResistance(branch,connectivity,branch_resistances):
	global resistance
	global parallel_res
	#checks to see if parent branch is in connectivity dictionary
	if branch in connectivity.keys() and len(connectivity[branch])>0:
		#If yes, then loop through each child branch and recursively call the method for downstream branches summed in parallel
		parallel_res = 0
		for child_branch in connectivity[branch]:
			parallel_res += 1.0/getResistance(child_branch,connectivity,branch_resistances)
		resistance = 1.0/parallel_res + branch_resistances[branch]

	else:
		#Otherwise, return the resistance of the branch itself
		return branch_resistances[branch]
	print({branch:resistance})
	return resistance

# Recursively find the orders of all segments in model
def findStrahOrders(segOrder, parentSeg, maxGen):
	d_orders = []

	# Track max generation number in model: each bifurcation adds 1 generation definition
	gen_num = []

	# Find Daughter Orders
	#print(jointSeg[parentSeg])
	for daughterSeg in jointSeg[parentSeg]: # find daughter segments of parent seg
		# If daughter not defined, make it the new parent to find orders below
		if segOrder[daughterSeg] == 0: 
			ordersNotDefined = True
			
			[segOrder, maxGen] = findStrahOrders(segOrder, daughterSeg, maxGen) # recursively call down the tree
			gen_num.append(maxGen)

			# Once all daughter orders found, backs out and can now define current daughter's order and append
			d_orders.append(segOrder[daughterSeg])
		else: # Append daughter's order to list of daughter orders
			d_orders.append(segOrder[daughterSeg])
			maxGen = 0
			gen_num.append(maxGen)
	
	# Determine the Parent Order based on daughter order
	#print('segOrder=',len(segOrder))
	#print('parentSeg=',parentSeg)
	#print('d_orders=',len(d_orders))
	if len(set(d_orders)) == 1: # if all daughters' orders the same, increase parent order by 1
		segOrder[int(parentSeg)] = d_orders[0]+1
	elif len(set(d_orders)) < 1:
		segOrder[int(parentSeg)] = 1
	else: # if daughter orders not the same, parent order is the largest daughter order
		segOrder[int(parentSeg)] = max(d_orders)

	# Only add generation at end of parent bifurcation for the largest gen #
	#maxGen = max(gen_num)
	maxGen += 1
	

	return(segOrder, maxGen)

# (B) Classify each segment into a diameter-based order from greatest to least
#	keep order if within 15% of the greatest diameter in that order
def initLargestDiam(huangSegmentDiameters, huangSegOrder, segmentsInHuangSegments):
	sortedDiameters = sorted(huangSegmentDiameters.items(), key=lambda x: x[1], reverse=True) #sort diameters from largest to smallest
	currOrder = 15
	maxOrder = 15
	isFirst = True
	smallestOrder = 0
	greatestDiameter = 0 
	for segment in sortedDiameters:
		if isFirst:
			huangSegOrder[segment[0]] = currOrder
			greatestDiameter = segment[1]
			isFirst = False
			continue 
		if greatestDiameter*0.85 <= segment[1]:
			huangSegOrder[segment[0]] = currOrder
		else:
			currOrder -= 1
			huangSegOrder[segment[0]] = currOrder
			greatestDiameter = segment[1]
			smallestOrder = currOrder


	if smallestOrder < 1:
		add = 1 - smallestOrder
		for segment in huangSegOrder:
			huangSegOrder[segment] += add


	currOrder = 15
	finished = False
	counter = 0
	while not finished:
		counter += 1
		finished = True
		for huangSegment in range(0, len(huangSegmentDiameters)):
			if huangSegment == 0:
				huangSegOrder[huangSegment] = currOrder
				continue
			# firstSeg = segmentsInHuangSegments.get(huangSegment)[0]
			parentSVSeg = 0
			for inletSeg in jointSeg:
				for outletSeg in jointSeg.get(inletSeg):
					if int(huangSegment) == int(outletSeg):
						parentSVSeg = inletSeg
			parentHuangSeg = 0
			for huangSeg in segmentsInHuangSegments:
				for seg in segmentsInHuangSegments.get(huangSeg):
					if int(seg) == int(parentSVSeg):
						parentHuangSeg = huangSeg
						break
			parentDiameter = huangSegmentDiameters.get(parentHuangSeg)
			if parentDiameter*.85 < huangSegmentDiameters.get(huangSegment) and parentDiameter*1.15 > huangSegmentDiameters.get(huangSegment):
				if huangSegOrder[parentHuangSeg] > 0:
					huangSegOrder[huangSegment] = huangSegOrder.get(parentHuangSeg)
				else:
					finished = False
			elif huangSegmentDiameters.get(huangSegment) > parentDiameter * 1.15:
				if huangSegOrder[parentHuangSeg] > 0:
					huangSegOrder[huangSegment] = huangSegOrder.get(parentHuangSeg) + 1
				else:
					finished = False
			else:
				if huangSegOrder[parentHuangSeg] > 0:
					huangSegOrder[huangSegment] = huangSegOrder.get(parentHuangSeg) - 1
				else:
					finished = False

	return(huangSegOrder)

# Find segmental arteries and scale orders to order 14 accordingly
# MODIFIED 11/12/18: No segmental ordering, only force MPA to have order 16
def orderSegmentals(segDiam, segOrder, inletSeg):
	# Segmental Arteries = largest arteries branching from LPA/RPA vessel
	count = 0

	#print(segOrder)

	# Force inlet to be Order 16 and translate all downstream orders
	#if not( ' ' in segmentalParents): # both left/right included, so MPA included
	print("INLET SEG: " + str(inletSeg) + " " + str(segOrder[int(inletSeg)]))
	print(jointSeg[inletSeg])
	segOrder[int(inletSeg)] = 11
	print('CA')
	for d_seg in jointSeg[inletSeg]:
		if segOrder[d_seg] != 11-1:
			segOrder[d_seg] = 11-1
			orderTransl = -1
			print(d_seg)
			print(orderTransl)
			segOrder = daughterOrders(segOrder, d_seg, orderTransl)

	
	return(segOrder)

# Recursively change all orders of segmental arteries
def daughterOrders(segOrder, p_seg, orderTransl):
	d_segs = jointSeg[str(p_seg)]
	for d_seg in d_segs:
		segOrder[d_seg] = segOrder[d_seg] + orderTransl
		segOrder = daughterOrders(segOrder, d_seg, orderTransl)
		# print("Downstream Segmentals {}: Order b4 {}, Order after {}".format(d_seg, segOrder[d_seg]-orderTransl, segOrder[d_seg]))

	return(segOrder)

def ConnectivityMatrix(segBifurcation,segName,segOrder,maxOrder,inletSegName):
	# CONNECTIVITY MATRIX
	# Define elements as vessels of the same order connected in series
	vesElem = defaultdict(list) #{Elem #: [seg #'s in element]} <- segments in element should all be the same order
	p_segName = inletSegName
	elem_segs = [p_segName] #keep track of segments sorted into an element
	tmp_elemNum = 0
	p_segNum = segName[p_segName]
	p_segOrd = segOrder[int(p_segNum)]
	vesElem[tmp_elemNum] = [int(p_segNum)]
	allSegsAdded =  all(elem in elem_segs  for elem in segName) #check to see if all segments added
	otherD_segs = [] #keep track of segments not yet sorted into an element, parse through this list when traversed down 1 vessel
	foundElemD = False

	# iterate through all segments until all segments added to an element group
	while not(allSegsAdded):
		# iterate through all daughter segments of parent segment
		d_segNames = copy.copy(segBifurcation[p_segName])
		#print(str(p_segName) + " " + str(p_segOrd) + " " + str(d_segNames))
		for d in d_segNames:
			d_segNum = segName[d]
			d_segOrd = segOrder[int(d_segNum)]
			#print("\t" + str(d) + " " + str(d_segNum) + " " + str(d_segOrd))

			# Check if daughter segment has same order as parent
			if d_segOrd == p_segOrd:
				vesElem[tmp_elemNum].append(int(d_segNum))
				elem_segs.append(d)
				d_segNames.remove(d)
				otherD_segs.extend(d_segNames)

				foundElemD = True
				p_segName = d
				p_segNum = segName[d]
				p_segOrd = segOrder[int(p_segNum)]
				break

		# Change parents if end of current element
		if not(foundElemD):
			if d_segNames == []:
				p_segName = otherD_segs[0]
				otherD_segs = otherD_segs[1:]
			else:
				d_segNames.remove(d)
				otherD_segs.extend(d_segNames)
				p_segName = d

			p_segNum = segName[p_segName]
			p_segOrd = segOrder[int(p_segNum)]
			tmp_elemNum += 1
			vesElem[tmp_elemNum] = [int(p_segNum)]
			elem_segs.append(p_segName)
			
		foundElemD = False
		allSegsAdded =  all(elem in elem_segs  for elem in segName)


	#Create Connectivity Matrix
	connectivityMatrix = np.zeros((maxOrder+4, maxOrder+4))
	for i in range(0,maxOrder+2): #Set first row and first column to be labeled -> first column = parent artery orders; first row = child artery orders
		connectivityMatrix[0][i] = i
		connectivityMatrix[i][0] = i

	# Iterate through all elements
	elementsPerOrder = defaultdict(list) # Order: number of elements of that order
	elemChildOrders = defaultdict(list) # Element #: [array of size 15 counting number of children segments of that order from element]
	#print(vesElem)
	#print(len(vesElem))
	for elem in vesElem:
		# Initialize count of child orders for that element
		elemChildOrders[elem] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

		# Count number of elements per order
		elem_ord = segOrder[vesElem[elem][0]]
		if elementsPerOrder[elem_ord] == []:
			elementsPerOrder[elem_ord] = 1
		else:
			elementsPerOrder[elem_ord] += 1

		# Add children to connectivity matrix for each element
		for seg in vesElem[elem]:
			for d in segBifurcation[seg]:
				if d not in vesElem[elem]:
					child_segOrd = segOrder[d]
					connectivityMatrix[child_segOrd][elem_ord] += 1

					elemChildOrders[elem][child_segOrd] += 1


	#print(connectivityMatrix)
	#print(elemChildOrders)
	sdConnectivityMatrix = copy.copy(connectivityMatrix)
	# Divides each element of the connectivity matrix by the number of elements in the parent order
	# **includes outlets in number of elements, but should not include outlet elements because continued
	for parentOrder in range(1,maxOrder+1):
		for childOrder in range(1,maxOrder+1):
			if elementsPerOrder.get(parentOrder, 1) != 0:
				connectivityMatrix[childOrder][parentOrder] /= elementsPerOrder.get(parentOrder, 1)

	#print("\nElements Per Order:\n" + str(elementsPerOrder))
	#print("\nSegments in each element:\n" + str(vesElem))
	#print("\nChild Order sin each element:\n" + str(elemChildOrders))
	#print(connectivityMatrix)

def DiamStrahler(segDiam,segQOI,segOrder,maxOrder):
	avgDiameters = defaultdict(float)	# {Order : Average Diameter}
	stdDev = defaultdict(list)			# {Order : Std Dev for Diameter}
	avgQOI = []
	stdDevQOI = []
	for i in range(0,len(segQOI)):
		avgQOI.append(defaultdict(float))
		stdDevQOI.append(defaultdict(list))
	sortedDiameters = sorted(segDiam.items(), key=lambda x:x[1]) # [ (str(seg #), float(diameter)), ...] smallest diam to largest
	sortedSeg = sorted(segOrder, key=segOrder.get) # [ seg #, ...] sorted from smallest to largest order by segment #
	
	print(len(sortedDiameters))
	print(len(sortedSeg))
	# Calculate average order's diameters using initial ordering classification scheme
	currOrder = 1
	currOrderSegments = []
	currOrderSegID = []
	currOrderQOI = []
	for i in range(0,len(segQOI)):
		currOrderQOI.append([])
	# print(segDiam)
	for i in sortedSeg: #Iterate through all Huang Segments -> sorted from smallest to largest order
		if(segOrder[i] == currOrder): #append diameters in current order to np array
			if isinstance(segDiam[i],float):
				currOrderSegments.append(segDiam[i])
			currOrderSegID.append(i)
			for q in range(0,len(segQOI)):
				if isinstance(segQOI[q][i],list) and len(segQOI[q][i])>0:
					print(segQOI[q][i][0])
					currOrderQOI[q].append((segQOI[q][i][0]+segQOI[q][i][1])/2)
				elif isinstance(segQOI[q][i],float): 
					currOrderQOI[q].append(segQOI[q][i])
				else:
					print('ERROR')
		else: #if segment not in current order, calculate the average diameters/stdev of all diameters in current order, move onto next order
			npCurrOrderSegments = np.asarray(currOrderSegments)
			npCurrOrderQOI = []
			for q in range(0,len(currOrderQOI)):
				npCurrOrderQOI.append(np.asarray(currOrderQOI[q]))
			if len(npCurrOrderSegments) >= 1: #ensures that array for current order is not empty
				avgDiameters[currOrder] = np.mean(npCurrOrderSegments)
				stdDev[currOrder] = np.std(npCurrOrderSegments) #stdev diameter
			for q in range(0,len(currOrderQOI)):
				if len(npCurrOrderQOI[q]) >= 1:
					avgQOI[q][currOrder] = np.mean(npCurrOrderQOI[q])
					stdDevQOI[q][currOrder] = np.std(npCurrOrderQOI[q])
			#Change current order to the order of the current segment that was different from previous
			currOrderSegments = []
			for q in range(0,len(currOrderQOI)):
				currOrderQOI[q] = []
			currOrder = segOrder[i]
			currOrderSegments.append(segDiam[i])
			currOrderSegID = [i]
			for q in range(0,len(segQOI)):
				if not isinstance(segQOI[q][i],float) and len(segQOI[q][i])>0:
					currOrderQOI[q].append((segQOI[q][i][0]+segQOI[q][i][1])/2)
				elif isinstance(segQOI[q][i],float):
					currOrderQOI[q].append(segQOI[q][i])

	# Edge case for last order in list
	npCurrOrderSegments = np.asarray(currOrderSegments)
	npCurrOrderQOI = []
	for q in range(0,len(currOrderQOI)):
		npCurrOrderQOI.append(np.asarray(currOrderQOI[q]))
	avgDiameters[currOrder] = np.mean(npCurrOrderSegments)
	stdDev[currOrder] = np.std(npCurrOrderSegments)
	for q in range(0,len(currOrderQOI)):
		if len(npCurrOrderQOI[q]) >= 1:
			avgQOI[q][currOrder] = np.mean(npCurrOrderQOI[q])
			stdDevQOI[q][currOrder] = np.std(npCurrOrderQOI[q])
	print(currOrder,currOrderSegID)
	previousAverageDiameters = avgDiameters.copy()
	previousStdDevs = stdDev.copy()	

	for ord in avgDiameters:
		QOI_string = ''
		for q in range(0,len(avgQOI)):
			QOI_string += '\t' + str(avgQOI[q][ord]) + '\t' + str(stdDevQOI[q][ord])
		print((str(ord) + '\t' + str(avgDiameters[ord]) + '\t' + str(stdDev[ord]) + QOI_string + '\n'))
	 			
	# Repeat classification based on new averages and standard deviation of order diameter
	#	Segment maintain order if:
	#		(D_(n-1) + SD_(n-1) + D_n - SD_n)/2 < Di < (D_n + SD_n + D_(n+1) - SD_(n+1))/2
	#			D_(n-1) = average diameter of order one less than current segment's order
	#			SD_(n-1) = standard deviation of order one less than current segment's order
	#			D_n = average diameter of current segment's order
	#			SD_n = standard deviation of current segment's order
	#			D_(n+1) = average diameter of order one greater than current segment's order
	#			SD_(n+1) = standard deviation of order one greater than current segment's order
	# Continue iterating until change in diameter and change in standard deviation are less than 1%
	numInOrder = dict()
	finished = False
	while not finished:
		orderChange = True
		currOrder = 6
		for segment in segOrder: #iterate through all Huang Segments
			currOrder = segOrder[segment]
			if currOrder > 1: #Check if Huang segment's diameter is smaller than the lower bound cutoff
				#if segment == 0:
				if ((avgDiameters.get(int(currOrder-1),0) + stdDev.get(int(currOrder-1),0) + avgDiameters.get(int(currOrder)) - stdDev.get(int(currOrder)))/2 > segDiam[segment]):
					segOrder[segment] = currOrder - 1
					orderChange = False
				
			if currOrder < maxOrder: #Check if Huang segment's diameter greater than the upper bound cutoff
				if ((avgDiameters.get(int(currOrder)) + stdDev.get(int(currOrder)) + avgDiameters.get(int(currOrder+1), 0) - stdDev.get(int(currOrder+1), 0)) < segDiam[segment]):
					segOrder[segment] = currOrder + 1
					orderChange = False
					# if segOrder.get(segment) > maxOrder: #If order exceeds the max order, force the order to be the max Order
					# 	segOrder[segment] = maxOrder


		# Recalculate the average diameters and standard deviation of diameters in each order
		currOrder = 1
		currOrderSegments = []
		currOrderQOI = []
		for i in range(0,len(segQOI)):
			currOrderQOI.append([])
		currOrderSegID = []
		sortedSegOrder = sorted(segOrder.items(), key=lambda x:x[1])
		orders = []
		avgDiameters.clear()
		for i in range(0,len(segQOI)):
			avgQOI[i].clear()
		for segment in sortedSegOrder: #Iterate through all Huang Segments -> sorted from smallest diameter to largest
			orders.append(currOrder)
			if(segOrder[segment[0]] == currOrder): #append diameters in current to np array
				if isinstance(segDiam[segment[0]],float):
					currOrderSegments.append(segDiam[segment[0]])
				for q in range(0,len(segQOI)):
					if isinstance(segQOI[q][segment[0]],list) and len(segQOI[q][segment[0]])>1:
						currOrderQOI[q].append((segQOI[q][segment[0]][0]+segQOI[q][segment[0]][1])/2)
					elif isinstance(segQOI[q][segment[0]],float):
						currOrderQOI[q].append(segQOI[q][segment[0]])
			else: #if segment not in current order, calculate the average diameters/stdev of all diameters in current order
				npCurrOrderSegments = np.asarray(currOrderSegments)
				currOrderSegID.append(segment[0])
				npCurrOrderQOI = []
				for q in range(0,len(currOrderQOI)):
					npCurrOrderQOI.append(np.asarray(currOrderQOI[q]))
				if len(npCurrOrderSegments) >= 1: #ensures that array for current order is not empty
					avgDiameters[currOrder] = np.mean(npCurrOrderSegments) #avg diameters
					stdDev[currOrder] = np.std(npCurrOrderSegments) #stdev diameter
					numInOrder[currOrder] = len(npCurrOrderSegments)
				for q in range(0,len(currOrderQOI)):
					if len(npCurrOrderQOI[q]) >= 1:
						avgQOI[q][currOrder] = np.mean(npCurrOrderQOI[q])
						stdDevQOI[q][currOrder] = np.std(npCurrOrderQOI[q])
				currOrderSegments = []
				for q in range(0,len(currOrderQOI)):
					currOrderQOI[q] = []
				currOrder = segOrder[segment[0]] #change current order to the order of the current segment that was different from previous
				currOrderSegments.append(segDiam[segment[0]])
				for q in range(0,len(segQOI)):
					if not isinstance(segQOI[q][segment[0]],float) and len(segQOI[q][segment[0]])>1:
						currOrderQOI[q].append((segQOI[q][segment[0]][0]+segQOI[q][segment[0]][1])/2)
					elif isinstance(segQOI[q][i],float):
						currOrderQOI[q].append(segQOI[q][segment[0]])

		npCurrOrderSegments = np.asarray(currOrderSegments)
		npCurrOrderQOI = []
		for q in range(0,len(currOrderQOI)):
			npCurrOrderQOI.append(np.asarray(currOrderQOI[q]))
		avgDiameters[currOrder] = np.mean(npCurrOrderSegments)
		stdDev[currOrder] = np.std(npCurrOrderSegments)
		numInOrder[currOrder] = len(npCurrOrderSegments)
		for q in range(0,len(currOrderQOI)):
			if len(npCurrOrderQOI[q]) >= 1:
				avgQOI[q][currOrder] = np.mean(npCurrOrderQOI[q])
				stdDevQOI[q][currOrder] = np.std(npCurrOrderQOI[q])
		#Check if change in diameter and change in standard deviation are less than 1%
		diameterWithin1Percent = True
		stdDevWithin1Percent = True
		for order in range(1, maxOrder+1): #Iterate through all orders
			divBy = 1
			if previousAverageDiameters.get(order, 0) != 0:
				divBy = previousAverageDiameters.get(order, 0)
			if (np.abs(previousAverageDiameters.get(order, 0) - avgDiameters.get(order, 0)))/divBy > .00001:
				diameterWithin1Percent = False
				break
			divBy = 1
			if(previousStdDevs.get(order, 0) != 0):
				divBy = previousStdDevs.get(order, 0)
			if (np.abs(previousStdDevs.get(order, 0) - stdDev.get(order, 0)))/divBy > .00001:
				stdDevWithin1Percent = False
				break

		if diameterWithin1Percent and stdDevWithin1Percent and orderChange:
			finished = True


		previousAverageDiameters = avgDiameters.copy()
		previousStdDevs = stdDev.copy()
	return avgDiameters,stdDev, avgQOI, stdDevQOI, numInOrder,segOrder

def main():

if __name__ == '__main__':
	main()