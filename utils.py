import os
import dicom
import numpy as np
from collections import Counter
import itertools

import scipy.misc
import re

import json
params = json.load(open('SETTINGS.json'))

def count_dirs(directory):
    while True:
        try:
            subdirs = next(os.walk(directory))[1]
        except StopIteration:
            return 0
        if len(subdirs) == 1:
            directory = os.path.join(directory, subdirs[0])
        else:
            break
    return len(subdirs)

Nt = count_dirs(params['TRAIN_DATA_PATH'])
if not Nt: Nt = 500
if Nt != 500: print 'wrong number of training studies %d . Did you mixed with validation?'%Nt
Nv = count_dirs(params['VALID_DATA_PATH'])
if not Nv: Nv = 200
if Nv != 200: print 'wrong number of validation studies %d. Did you mixed with validation?'%Nv
Ns = count_dirs(params['TEST_DATA_PATH'])
if not Ns: Ns = 440
if Ns != 440: print 'wrong number of test studies %d'%Ns


out_dir = params['OUT_DATA_PATH']
temp_dir = params['TEMP_DATA_PATH']

import hashlib
import subprocess
def awscp(fn, upload=False, verbose=False):
    if not fn:
        return
    local_fn = os.path.join(temp_dir,fn)
    remote_fn = os.path.join(out_dir,fn)
    fns = [local_fn, remote_fn] if upload else [remote_fn, local_fn]
    # if not os.path.isfile(local_fn) or os.stat(local_fn).st_size < 1:
    if remote_fn.startswith('s3://'):
        cmd = 'aws s3 cp'
    else:
        cmd = 'cp'
    cmd += ' ' + ' '.join(fns)
    if verbose:
        print cmd
    subprocess.call(cmd.split())
    if verbose:
        with open(local_fn,'r') as fp:
            print fn, hashlib.sha224(fp.read()).hexdigest()

# see 160224-segment-test
avg_image_orientations = np.array([ 0.61659826,  0.77685789, -0.12766521, -0.39722351,  0.16699159,
       -0.90234914])

class Dataset(object):
    def __init__(self, subdir, prefix='sax', verbose=0):
        """
        prefix: what kind of slices to read: 'sax', '2ch' or '4ch'
        """
        self.name = subdir
        self.verbose = verbose
        if subdir <= 0:
            raise Exception('bad study number %d. Study numbers begin with 1.', subdir)
        elif subdir <= Nt:
            directory = os.path.join(params['TRAIN_DATA_PATH'], str(subdir))
        elif subdir <= Nt+Nv:
            directory = os.path.join(params['VALID_DATA_PATH'], str(subdir))
        elif subdir <= Nt+Nv+Ns:
            directory = os.path.join(params['TEST_DATA_PATH'], str(subdir))
        else:
            raise Exception('study number %d too big. Did you made test data available?', subdir)

        self.prefix = prefix
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = []
        for s in subdirs:
            m = re.match("%s_(\d+)"%self.prefix, s)
            if m is not None:
                slices.append(int(m.group(1)))
        slices = sorted(slices)

        # build slice_map. For each slice find the offset which is the
        # index used to describe all frames in the slice and if there
        # are multiple repeated scans, find the index of the maximal scan
        # repeat
        slices_map = {}
        times = None
        maxscan = None
        extra_slices = []
        for s in slices:
            files = next(os.walk(os.path.join(directory, "%s_%d" % (self.prefix,s))))[2]
            offset = None
            scan_times = {}

            for f in files:
                m = re.match("IM-(\d{4,})-(\d{4})(-\d{4})?\.dcm", f)
                if m is not None:
                    t = int(m.group(2))
                    if offset is None:
                        offset = int(m.group(1))
                    else:
                        assert offset == int(m.group(1))
                    if m.group(3) is not None:
                        scan = int(m.group(3)[1:])
                    else:
                        scan = -1
                    if scan in scan_times:
                        scan_times[scan].append(t)
                    else:
                        scan_times[scan] = [t]
            if -1 in scan_times:
                assert len(scan_times) == 1
                maxscan = None
                if times is None:
                    times = set(scan_times[-1])
                else:
                    times |= set(scan_times[-1])
            else:
#                for v in scan_times.values():
#                    assert len(set(v)) == len(set(times))
                maxscan = max(scan_times.keys())
                if times is None:
                    times = set(sum(scan_times.values(),[]))
                else:
                    times |= set(sum(scan_times.values(),[]))

            slices_map[s] = (offset, maxscan)
            if maxscan is not None and maxscan > 2:
                if self.verbose:
                    print self.name,"expanding slice %d"%s,maxscan,scan_times.keys()
                for scan in scan_times.keys():
                    if scan == maxscan:
                        continue
                    assert scan != 0
                    s1 = s + scan/100.
                    extra_slices.append(s1)
                    slices_map[s1] = (offset, scan)


        self.directory = directory
        if times is not None:
            self.time = sorted(times)
        else:
            self.time = [0]
        self.slices = sorted(slices+extra_slices)
        self.slices_map = slices_map
        if maxscan is not None:
            if self.verbose:
                print self.name, 'maxscan', maxscan
        self.PatientAge = None
        self.PatientSex = None
        self.PatientAngle = None
        self.outS = None # wanted square size of images
        self.PatientPosition = None
        self.ImageOrientationPatient = None

        for t in ['TriggerTime', 'NominalInterval', 'RepetitionTime']:
            setattr(self,'mean'+t, np.nan)
            setattr(self,'var'+t, np.nan)
        self.shapes = []
        self.shape = (0,0,0,0)
        self.area = np.nan
        self.dist = np.nan
        self.images = np.empty(self.shape)


    def _filename(self, s, t):
        offset, maxscan = self.slices_map[s]
        if maxscan is None:
            return os.path.join(self.directory,"%s_%d" % (self.prefix,s), "IM-%04d-%04d.dcm" % (offset, t))
        else:
            sint = int(100.*(s - int(s)) + 1e-3)
            if sint == 0:
                return os.path.join(self.directory,"%s_%d" % (self.prefix,s), "IM-%04d-%04d-%04d.dcm" % (offset, t, maxscan))
            else:
                return os.path.join(self.directory,"%s_%d" % (self.prefix,s), "IM-%04d-%04d-%04d.dcm" % (offset, t, sint))

    def _read_dicom(self, f):
        try:
            return dicom.read_file(f)
        except:
            if self.verbose:
                print 'cant read',f
            return None

    def _read_dicom_image(self, d):
        """
        use self.S to scale image. If < 0 make it square

        :param d: dicom image
        :return: image, area scale
        """
        if d is None:
            return self.last_image, self.last_area_scale
        img = d.pixel_array
        img = np.array(img)

        H,W = img.shape
        HW = max(H,W)
        S = self.outS
        if S is not None and S < 0:
            S = HW
        area_scale = 1.
        if (S is not None) and ((S != H) or (S != W)):
            ox = (HW - H)//2
            oy = (HW - W)//2
            f = np.zeros((HW,HW))
            f[ox:ox+H,oy:oy+W] = img
            img = scipy.misc.imresize(f, (S,S))
            area_scale = (float(HW)/S) * (float(HW)/S)

        self.last_image = img
        self.last_area_scale = area_scale
        return img, area_scale

    def age(self,x):
        if x is None:
            return
        x = x.PatientAge
        if x.endswith('Y'):
            x = float(x[:-1])
        elif x.endswith('M'):
            x = float(x[:-1])/12.
        else:
            x = float(x[:-1])/54.
        if self.PatientAge is None:
            self.PatientAge = x
        else:
            assert self.PatientAge == x

    def sex(self,x):
        if x is None:
            return
        x = x.PatientSex
#         x = x.split()[1]
        x = {'M':1, 'F':0}[x]
        if self.PatientSex is None:
            self.PatientSex = x
        else:
            assert self.PatientSex == x

    def angle(self, d):
        if d is None:
            return
        image_orientation = np.array(d.ImageOrientationPatient)
        y = np.dot(image_orientation[:3], avg_image_orientations[:3])
        x = np.dot(image_orientation[:3], avg_image_orientations[3:])
        angle = np.arctan2(y, x) / np.pi * 180 - 75
        if self.PatientAngle is None:
            self.PatientAngle = angle
#        else:
#            if np.around(self.PatientAngle,1) != np.around(angle,1):
#                print 'angle change','%f %f'%(self.PatientAngle,angle)

    def patient_position(self,x):
        """
        1.Head First-Prone
        2.Head First-Supine
        3.Head First-Decubitus Right
        4.Head First-Decubitus Left
        5.Feet First-Decubitus Left
        6.Feet First-Decubitus Right
        7.Feet First-Prone
        8.Feet First-Supine.
        Definitions:

        Head First means the patient was laying on the imaging couch with the head facing the imaging device first.

        Feet first means the patient was laying on the imaging couch with the feet facing the imaging device first.

        Prone means the patient is laying on his/her stomach. (Patient's face being positioned in a downwards (gravity) direction.)

        Supine means the patient is laying on his/her back. (Patient's face being in an upwards direction.)

        Decubitus Right means the patient is laying with his/her right side in a downwards direction.

        Decubitus Left means the patient is laying with his/her left side in a downwards direction.
        """
        if x is None:
            return
        x = x.PatientPosition
        if self.PatientPosition is None:
            self.PatientPosition = x
            if self.PatientPosition != 'HFS':
                if self.verbose:
                    print self.name,self.PatientPosition
        else:
            assert self.PatientPosition == x

    def _read_all_dicom_images(self):
        dicoms = [[self._read_dicom(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices]

        if all([d is None for d in itertools.chain(*dicoms)]):
            return
        self.dicoms = dicoms

        # immutable fields
        map(self.sex, itertools.chain(*dicoms))
        map(self.angle, itertools.chain(*dicoms))
        map(self.age, itertools.chain(*dicoms))
        map(self.patient_position, itertools.chain(*dicoms))

        # mutable fields
        for t in ['TriggerTime', 'NominalInterval', 'RepetitionTime']:
            T = np.array([[dicoms[d][i].get(t) if dicoms[d][i] is not None else np.nan
                                     for i,_ in enumerate(self.time)]
                                    for d,_ in enumerate(self.slices)])
            if t == 'TriggerTime':
                T = T[:,1:]-T[:,:-1]
            setattr(self,'mean'+t, np.nanmean(T,axis=-1))
            setattr(self,'var'+t, np.nanvar(T,axis=-1))


        # make sure all images from the same slice have the same orientation
        ImageOrientationPatient = dicoms[0][0].ImageOrientationPatient
        for dd,d in enumerate(dicoms):
            image_orientation = d[0].ImageOrientationPatient
            for tt,t in enumerate(d):
                if t is not None:
                    assert image_orientation == t.ImageOrientationPatient

        S = len(dicoms)

        h = Counter(tuple(d.ImageOrientationPatient) for d in itertools.chain(*dicoms) if d is not None)
        self.ImageOrientationPatient = h.most_common(1)[0][0]
        h = Counter(tuple(np.around(np.cross(d.ImageOrientationPatient[:3],d.ImageOrientationPatient[3:]),3))
                    for d in itertools.chain(*dicoms) if d is not None)
        self.image_direction = h.most_common(1)[0][0]
        idxs = filter(lambda s: tuple(np.around(np.cross(dicoms[s][0].ImageOrientationPatient[:3],
                                                        dicoms[s][0].ImageOrientationPatient[3:]),3)) == self.image_direction,
                      range(S))
        if len(idxs) < S:
            if self.verbose:
                print self.name,"lost %d on image direction"%(S-len(idxs))

        images = [[self._read_dicom_image(dicoms[d][i])
                                 for i,_ in enumerate(self.time)]
                                for d,_ in enumerate(self.slices)]

        # take out the area scale of each image
        area_scales = [map(lambda x: x[1], slice) for slice in images]
        # make sure it is the same for all images from the same slice
        assert all(len(set(list(slice))) == 1 for slice in area_scales)
        # keep one example per slice
        area_scales = [slice[0] for slice in area_scales]

        images = [map(lambda x: x[0], slice) for slice in images]

        self.shapes = Counter(image.shape for image in itertools.chain(*images))
        self.shape = self.shapes.most_common(1)[0][0]
        assert S == len(images)
        T = len(images[0])
        if not all(len(slice) == T for slice in images):
            if self.verbose:
                print "IMAGES: not the same time"
        if not all([all([slice[0].shape == s.shape for s in slice]) for slice in images]):
            if self.verbose:
                print "IMAGES: not the same shape in a slice"

        n = len(idxs)
        idxs = filter(lambda s: images[s][0].shape == self.shape, idxs)
        if len(idxs) < n:
            if self.verbose:
                print self.name,"lost %d on shape"%(n - len(idxs))

        areas = []
        xxs = []
        for dd in idxs:
            d = dicoms[dd]
            area_scale = area_scales[dd]
            slice_area = None
            for tt,t in enumerate(d):
                if t is None:
                    continue
                x1,y1 = map(float,t.PixelSpacing)
                np.testing.assert_approx_equal(x1,y1, err_msg="not square")
#                 x1 = np.around(x1,2)
                area = np.around(x1*y1*area_scale,5)
                if slice_area is None:
                    slice_area = area
                else:
                    np.testing.assert_approx_equal(slice_area,area, err_msg="slice area not square")
                areas.append(area)
                xxs.append((area,dd,tt))
        self.area = Counter(areas).most_common(1)[0][0]
        self.xxs = filter(lambda x: np.around(x[0],4)!=np.around(self.area,4),xxs)
        bad_slices = set(map(lambda x: x[1], self.xxs))


        # if self.xxs:
        #     print self.name, self.xxs
        idxs = sorted(set(idxs) - bad_slices)
        if bad_slices:
            if self.verbose:
                print self.name, "lost %d on bad pixel spacing"%(len(bad_slices))


        # make sure all images from the same slice have the same location
        self.SliceLocation = dicoms[0][0].SliceLocation
        for dd,d in enumerate(dicoms):
            loc = d[0].SliceLocation
            for tt,t in enumerate(d):
                if t is not None:
                    np.testing.assert_approx_equal(t.SliceLocation,loc)

        # make sure all images, from same slice, have same image position
        self.ImagePositionPatient = np.array(dicoms[0][0].ImagePositionPatient)
        for d in dicoms:
            for t in d:
                if t is not None:
                    assert d[0].ImagePositionPatient == t.ImagePositionPatient

        slice_locations = np.array([np.dot(d[0].ImagePositionPatient, self.image_direction) for d in dicoms])
        slice_centers = []
        slice_center_direction = []
        for d in dicoms:
            q = d[0]
            image_center2D = q.PixelSpacing * (np.array([q.Columns,q.Rows])-np.ones(2))/2.
            image_center3D = np.dot(image_center2D, np.reshape(q.ImageOrientationPatient,(2,3)))
            center =  q.ImagePositionPatient + image_center3D
            slice_centers.append(center)
            direction = np.argmax(np.abs(np.cross(q.ImageOrientationPatient[:3],q.ImageOrientationPatient[3:])))
            slice_center_direction.append(center[direction])
        slice_centers = np.array(slice_centers)
        # What you see in the "Location" field in Osirix is (and good only knows why):
        # print self.name,' '.join(map(str,np.around(slice_center_direction,2)))
        self.slice_centers = slice_centers
        self.slice_locations = slice_locations

        self.dist = np.nan
        self.slice_location_range = max(slice_locations) - min(slice_locations)
        if len(dicoms) > 1:
            locidx = np.argsort(slice_locations)
            slice_locations = slice_locations[locidx]

            # find the most popular distance between slices of distances 1 or 2
            distances = np.hstack([slice_locations[d:]-slice_locations[:-d] for d in range(1,5)])
            # allow for errors of +/- 0.04
            h = Counter(np.around(np.around(distances*4,2)/4,2))
            maxh = max(h.values())
            self.dist = min([k for k,v in h.items() if v==maxh])
            # find the longest sequence of slices with the most popular distance
            sequences = []
            for i in range(len(slice_locations)):
                if locidx[i] not in idxs:
                    continue
                if any(i in sequence for sequence in sequences):
                    continue
                sequence = [i]
                sequences.append(sequence)
                j = i+1
                while j < len(slice_locations):
                    ddx = [k for k in range(j,len(slice_locations))
                           if locidx[k] in idxs and not any(k in sequence for sequence in sequences)]
                    if not ddx:
                        break
                    dd = np.abs(self.dist - (slice_locations[ddx] - slice_locations[sequence[-1]]))
                    ddi = ddx[np.argmin(dd)]
                    ddm = np.min(dd)
                    if ddm < 0.02:
                        sequence.append(ddi)
                        j = ddi + 1
                    else:
                        break

            # if len(sequences) > 1:
            #     print self.name, 'seqs',[[self.slices[locidx[s]] for s in sequence] for sequence in sequences]
            maxseq = max(map(len,sequences))  # length of each sequence
            maxseq = filter(lambda x: len(x)==maxseq, sequences)[0]
            self.slice_location_range = slice_locations[maxseq[-1]] - slice_locations[maxseq[0]]

            # convert back to slice index
            maxseq = [locidx[s] for s in maxseq]
            # find new clean set of slices
            n = len(idxs)
            idxs = sorted(set(idxs) & set(maxseq))
            if len(idxs) < n:
                if self.verbose:
                    print self.name,"lost %d on not in sequence"%(n-len(idxs))

        # re-order image according to location
        idxs = np.array(idxs)
        idxs = idxs[np.argsort([dicoms[i][0].SliceLocation for i in idxs])]
        images = [images[s] for s in idxs]
        self.images = np.array(images)
        self.islice_map = [(self.slices[s],self.slices_map[self.slices[s]]) for s in idxs]
#         if not max(idxs)-min(idxs)==len(idxs)-1:
#             print self.name, "IMAGES: not contigous"

    def load(self, S=None, Z=None):
        """
        Read images and DICOM attributes.

        Single value DICOM attributes:
        ImageOrientationPatient, PatientPosition, PatientAge, PatientSex
        variable DICOM attribute compute mean and var:
        'TriggerTime', 'NominalInterval', 'RepetitionTime'


        S: int
            Wanted square size of images. Images are zero padded to a sqare
            and then resized to S. If S < 0 then just make the images square if they are not
        Z: int
            Wanted number of frames. Zero padd missing frames
        """
        self.outS = S
        self._read_all_dicom_images()
        N,F,H,W = self.images.shape
#         print N,F,H,W
        if ((Z is not None) and (Z != F)):
            # Zero padd missing frames
            images = np.empty((N,Z,H,W))
            oz = (Z-F)//2
            for i, slice in enumerate(self.images):
                if oz > 0:
                    images[i,:oz,:,:] = slice[0]
                for j, frame in enumerate(slice):
                    images[i,j+oz,:,:] = frame
                if oz+len(slice) < Z:
                    images[i,oz+len(slice):,:,:] = slice[-1]
            self.images = images


# Utilitiy to convert array of images to GIF
import tempfile
import shutil
import base64
import os
import scipy.misc

def make_gif(frames, out_file="demo.gif"):
    d = tempfile.mkdtemp()
    frame_files = []
    for i, frame in enumerate(frames):
        file_name = os.path.join(d, "frame_%06d.png" % (i,))
        scipy.misc.imsave(file_name, frame)
        frame_files.append(file_name)
    x = subprocess.check_output(["convert", "-delay", "15", "-loop", "0",
                                 '*.png', out_file], cwd=d)
    shutil.move(os.path.join(d, out_file), out_file)
    shutil.rmtree(d)

if __name__ == "__main__":
    pass
