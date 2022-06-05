import numpy as np
import cv2
import os
from PIL import Image
import random

# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[3] == 1 or data.shape[3] == 3)
    all_stripe = []
    for i in range(int(data.shape[0]/per_row)):
        stripe = data[i*per_row]
        for k in range(i*per_row+1, i*per_row+per_row):
            stripe = np.concatenate((stripe, data[k]), axis=0)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=1)
    return totimg

# visualize image(as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert(len(data.shape) == 3)  # height*width*channels

    if data.shape[2] == 1:    # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))

    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))    # the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8)) # the image is between 0-1

    img.save(filename+'.png')
    return img

def read_image_and_name(path):
    imgdir = os.listdir(path)
    if "CHASEDB1" in path:
        imgdir.sort(key=lambda x: int(x[6:8]))  # do the sorting according to file names
    elif "IOSTAR" in path:
        imgdir.sort(key=lambda x: int(x[5:7]))  # do the sorting according to file names
    else:
        imgdir.sort(key=lambda x: int(x[0:2]))  # do the sorting according to file names
    imglst = []
    imgs = []
    for v in imgdir:
        imglst.append(path + v)
        imgs.append(cv2.imread(path + v))
    print(imglst)
    print('original images shape: ' + str(np.array(imgs).shape))
    return imglst,imgs


def read_label_and_name(path):
    labeldir = os.listdir(path)
    labellst = []
    labels = []
    if "CHASEDB1" in path:
        labeldir.sort(key=lambda x: int(x[6:8]))  # do the sorting according to file names
    elif "IOSTAR" in path:
        labeldir.sort(key=lambda x: int(x[5:7]))  # do the sorting according to file names
    else:
        labeldir.sort(key=lambda x: int(x[0:2]))  # do the sorting according to file names
    for v in labeldir:
        labellst.append(path + v)
        label = Image.open(path + v)
        if "CHASEDB1" in path:
            label = label.convert('L')
            label = np.asarray(label)
            label = label.reshape((label.shape[0], label.shape[1]))
        else:
            label = np.asarray(label)
        labels.append(label)
    print(labellst)
    print('original labels shape: ' + str(np.array(labels).shape))
    return labellst,labels


def resize(imgs,resize_width, resize_height):
    img_resize = []
    for file in imgs:
        img_resize.append(cv2.resize(file, (resize_width, resize_height)))
    return img_resize

def resize2SquareKeepingAspectRation(imgs, size, interpolation):
    img_resize = []
    for file in imgs:
        height, width = file.shape[:2]
        c = None if len(file.shape) < 3 else file.shape[2]
        if height == width:
            return cv2.resize(file, (size, size), interpolation)
        if height > width:
            diff = height
        else:
            diff = width
        x_pos = int((diff - width)/2.)
        y_pos = int((diff - height)/2.)
        if c is None:
            mask = np.zeros((diff, diff), dtype=file.dtype)
            mask[y_pos:y_pos+height, x_pos:x_pos+width] = file[:height,:width]
        else:
            mask = np.zeros((diff, diff, c), dtype=file.dtype)
            mask[y_pos:y_pos + height, x_pos:x_pos + width, :] = file[:height, :width, :]
        img_resize.append(cv2.resize(mask, (size, size), interpolation))
    return img_resize


# crop N images with the resolution of 576 by 576 into 48 by 48
def crop(image,dx):
    list = []
    for i in range(image.shape[0]):
        for x in range(image.shape[1] // dx):
            # the list here has appended 20*12*12, so the returned shape is (2880,48,48)
            for y in range(image.shape[2] // dx):
                list.append(image[i,  y*dx: (y+1)*dx,  x*dx: (x+1)*dx])
    return np.array(list)

# extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random_uncertainty(full_imgs,full_masks,patch_h,patch_w, N_patches):
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 3)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    print("patches per full image: " + str(patch_per_img))
    patches = np.empty((full_imgs.shape[0], patch_per_img, patch_h, patch_w, full_imgs.shape[3]))
    patches_masks = np.empty((full_imgs.shape[0], patch_per_img, patch_h, patch_w))
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    seed = 7
    np.random.seed(seed)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        iter_tot = 0  # iter over the total number of patches (N_patches)
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2), img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2), img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            patch = full_imgs[i, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2), :]
            patch_mask = full_masks[i, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[i,iter_tot,:,:,:] = patch
            patches_masks[i,iter_tot,:,:] = patch_mask
            iter_tot += 1   # total
            k += 1  # per full_img
    return patches, patches_masks


# extract patches randomly in the full training images
#  -- Inside OR in full image
def extract_random(full_imgs,full_masks,patch_h,patch_w, N_patches):
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 3)  # 4D arrays
    assert (full_imgs.shape[3] == 1 or full_imgs.shape[3] == 3)  # check the channel is 1 or 3
    assert (full_imgs.shape[1] == full_masks.shape[1] and full_imgs.shape[2] == full_masks.shape[2])
    patches = np.empty((N_patches, patch_h, patch_w, full_imgs.shape[3]))
    patches_masks = np.empty((N_patches, patch_h, patch_w))
    img_h = full_imgs.shape[1]  # height of the full image
    img_w = full_imgs.shape[2]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches/full_imgs.shape[0])  #N_patches equally divided in the full images
    print("patches per full image: " +str(patch_per_img))
    iter_tot = 0   # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x_center = random.randint(0+int(patch_w/2), img_w-int(patch_w/2))
            # print "x_center " +str(x_center)
            y_center = random.randint(0+int(patch_h/2), img_h-int(patch_h/2))
            # print "y_center " +str(y_center)
            patch = full_imgs[i, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2), :]
            patch_mask = full_masks[i, y_center-int(patch_h/2):y_center+int(patch_h/2), x_center-int(patch_w/2):x_center+int(patch_w/2)]
            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1   # total
            k += 1  # per full_img
    return patches, patches_masks



# check if the patch is fully contained in the FOV
def is_patch_inside_FOV(x,y,img_w,img_h,patch_h, Mask_Radius):
    x_ = x - int(img_w/2) # origin (0,0) shifted to image center
    y_ = y - int(img_h/2)  # origin (0,0) shifted to image center
    R_inside = Mask_Radius - int(patch_h * np.sqrt(2.0) / 2.0) #radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_*x_)+(y_*y_))
    if radius < R_inside:
        return True
    else:
        return False

# Estimated output of network is converted to image subpatches
# Estimated output of network size=[Npatches, patch_height*patch_width, 2]
def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape)==3)  #3D array: (Npatches,height*width,2)
    assert (pred.shape[2]==2 )  #check if the classes are 2
    pred_images = np.empty((pred.shape[0],pred.shape[1]))  #(Npatches,height*width)
    if mode=="original":        # the probability output of the network
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i,pix]=pred[i,pix,1] # pred[:,:,0] is non-segmentation output, and pred[:,:,1] is segmentation output

    elif mode=="threshold":                      # network probability-thresholds output
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i,pix,1]>=0.5:
                    pred_images[i,pix]=1
                else:
                    pred_images[i,pix]=0
    else:
        print("mode " +str(mode) +" not recognized, it can be 'original' or 'threshold'")
        exit()
    # the output form is modified as (Npatches,1, patch_height, patch_width)
    pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch_height, patch_width))
    return pred_images