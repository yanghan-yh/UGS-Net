'''
Created by WangQL

6/27/2018

operation for xml files
get structure of xml files from LIDC
'''
try: 
  import xml.etree.cElementTree as ET 
except ImportError: 
  import xml.etree.ElementTree as ET 
import sys,os
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np

def getEdgeMap(file, zloc, nodule_list):

  tree = ET.parse(file) 
  root = tree.getroot() 

  try:
    rangelist = []
    Img=np.zeros((512,512))

    for onenoduleid in nodule_list:
      rangelist, Img = getEdgeMap_id(root, onenoduleid, zloc)
      if rangelist != None:
        break
      
    # scipy.misc.imsave(str(j) + '.png', Img)
    # print('flag: ', flag)
    signtemp = False

    lnglist = []
    latlist = []
    for i in range(len(rangelist)-1):
      lnglist.append(rangelist[i][0])
      latlist.append(rangelist[i][1])

    # print(nodule_list[0])
    # print(lnglist)
    # print(latlist)
    maxlng = max(lnglist)
    minlng = min(lnglist)
    maxlat = max(latlist)
    minlat = min(latlist)

    for a in range(minlng, maxlng):
      for b in range(minlat, maxlat):
        point = [a, b]
        if isPointinPolygon(point, rangelist, maxlng, minlng, maxlat, minlat):
          Img[b][a] = 1
  except:
    signtemp = True

  return Img, signtemp


def getEdgeMap_id(root, nodule_list, zloc):
  flag = True
  j=0
  coo2=[]
  sign = 0

  rangelist = []
  flag = True
  Img=np.zeros((512,512))
  for ResponseHeader in root:
    # print('child')
    # print(ResponseHeader.tag)
    if not flag:
      break
    for readingSession in ResponseHeader:
      if not flag:
        break
      # print('    ',readingSession.tag)
      for unblindedReadNodule in readingSession:
        if not flag:
          break
        # print('        ',unblindedReadNodule.tag)
        if unblindedReadNodule.tag == '{http://www.nih.gov}noduleID':
          id = unblindedReadNodule.text.strip()
          try:
            input_id = int(nodule_list)
            id = int(id)
          except:
            input_id = str(nodule_list)
            id = str(id)
          # print(id_i, id)
          if id == input_id:
            # print('here here')
            for unblindedReadNodule in readingSession:
              if  unblindedReadNodule.tag=='{http://www.nih.gov}roi':
                coo=[]
                hoo=[]
                hoo1=[]
                j+=1
                for roi in unblindedReadNodule:
                  if not flag:
                    break
                  if roi.tag == '{http://www.nih.gov}imageZposition':
                    if abs(float(zloc) - float(roi.text)) < 0.001:
                      # existsign = True
                      # print("float(zloc)")
                      # print(float(roi.text))
                      for roi in unblindedReadNodule:
                        if roi.tag=='{http://www.nih.gov}edgeMap':
                          n=0
                          coo1=[]
                          for edgeMap in roi: 
                              if n:
                                hoo.append(int(edgeMap.text))
                              else:
                                hoo1.append(int(edgeMap.text))
                              n=1
                              coo1.append(int(edgeMap.text))
                          coo.append(coo1)
                          coo2.append(coo1)
                      for i in range(len(coo)):
                        Img[coo[i][1]][coo[i][0]]=1
                      flag = False
                      rangelist = coo
  return rangelist, Img

def isPointinPolygon(point, rangelist, maxlng, minlng, maxlat, minlat):  #[[0,0],[1,1],[0,1],[0,0]] [1,0.8]

    if (point[0] > maxlng or point[0] < minlng or
        point[1] > maxlat or point[1] < minlat):
        return False
    count = 0
    point1 = rangelist[0]
    for i in range(1, len(rangelist)):
        point2 = rangelist[i]
        if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
            return False
        if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
            point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0])/(point2[1] - point1[1])
            if (point12lng == point[0]):
                return False
            if (point12lng < point[0]):
                count +=1
        point1 = point2
    if count%2 == 0:
        return False
    else:
        return True