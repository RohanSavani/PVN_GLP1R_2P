import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import figure
import pandas as pd
import os
from matplotlib.colors import ListedColormap
import skimage
import skimage.measure as measure
from skimage.feature import canny

#Draw Image of all ROIs based on suite2p
def draw_image(path,rois_on):
    f=np.load(os.path.join(path,"F.npy"))
    Fneu=np.load(os.path.join(path, "Fneu.npy"))
    iscell=np.load(os.path.join(path,"iscell.npy"))
    spikes=np.load(os.path.join(path,"spks.npy"))
    stat=np.load(os.path.join(path,"stat.npy"),allow_pickle=True)
    ops=np.load(os.path.join(path,"ops.npy"),allow_pickle=True).item()

    #Get list of 'valid' cells, i.e. ROIs that are cells.
    cellValid = iscell[:,0]
    np_ops = np.array(ops['meanImg'])
    len_np_ops = 512
    cellIDMap = np.zeros([len_np_ops,len_np_ops], dtype = int)
    validCellList = np.where(cellValid==1)
    validCellList = np.squeeze(validCellList)
    validCellList = list(validCellList)
    
    #Create mask to be used to create the image - all ROIs including overlap
    #Overlap is included because otherwise contour drawi   ng can get messed up.
    #This does not affect ROI detection as each ROI is only compared with those from other sessions.
    cellID=[]
    tmp = []
    for iCell in validCellList:
        tmp = iCell;
        cellID=cellID+[tmp]
        for n in range(0,np.size(stat[iCell]['ypix'])):
            cellIDMap[stat[tmp]['ypix'],stat[tmp]['xpix']] = iCell
    
    #Plot image of ROIs, to be used for comparisons.
    img = cellIDMap
    fig, ax = plt.subplots()
    ax.imshow(img)
    if rois_on == True:
        for i in range(0,len(validCellList)):
            ax.annotate(validCellList[i],xy = (stat[validCellList[i]]['med'][1],stat[validCellList[i]]['med'][0]))
    plt.title(path)
    ax.set_axis_on()
        
    return img,validCellList,stat,ops

#Draw contours based on images.
def draw_contours(img):
    
    contours = measure.find_contours(img, 0.0005)
    return contours

#Match the list of contours to each ROI (for plotting), based on their average y-value. 

def rois_matched_to_contours(stat,validCellList,contours):
    statsy_avg = [ [0]*2 for i in range(len(validCellList))]
    for i in range(0, len(validCellList)):
        tmp = int(np.mean(stat[i]['ypix']))
        statsy_avg[i] = [i, tmp]
        
    statsy_avg.sort(key=lambda x: x[1])
    contoury_avg = [ [0]*2 for i in range(len(contours))]
    for i in range(0,len(contours)):
        tmp = int(np.mean(contours[i][:, 0]))
        contoury_avg[i] = [i, tmp]
        
    contoury_avg.sort(key=lambda x:x[1])
    
    roi_numbers = [ [0]*2 for i in range(len(validCellList))]
    for i in range(0,len(validCellList)):
        roi_numbers[i][0] = statsy_avg[i][0]
        if len(contoury_avg) > i:
            roi_numbers[i][1] = contoury_avg[i][0]
        else:
            roi_numbers[i][1] = [0][0]
    return roi_numbers 

#Create arrays of all coordinate pairs within the first and second ROIs.
def coords(stat,cells):
    stats = [0,0]*len(cells)
    statsy = [0]*len(cells)
    statsx = [0]*len(cells)
    for i in range(0,len(cells)):
        statsy[i] = list(stat[cells[i]]['ypix'])
        statsx[i] = list(stat[cells[i]]['xpix'])
        stats[i] = [statsx[i],statsy[i]]
    stats_coords = [0]*len(cells)
    for i in range(0,len(cells)):
        stats[i][:] = [stats[i][:][0],stats[i][:][1]]
        stats_coords[i] = [0]*len(stats[i][0])
    for i in range(0,len(cells)):
        for j in range(0,len(stats[i][0])):
            stats_coords[i][j] = ([stats[i][0][j],stats[i][1][j]])
    return stats_coords

#Transform the currently analyzed set of ROIs to the baseline. 
#Checks each baseline ROI against every other analyzed ROI for overlap.
# Outputs a table with the fraction of pixels of a baseline ROI that overlap with every other 
# (for most this should be zero).
def roi_intersect(stats_1,stats_2,cells1,cells2,transform):
    stats1_coords = coords(stats_1,cells1)
    stats2_coords = coords(stats_2,cells2)
    
    #Transform each pixel within each ROI using our transformation function and re-convert to a list
    for i in range(0,len(cells2)):
        for j in range(0,len(stats2_coords[i])):
            stats2_coords[i][j] = transform(stats2_coords[i][j])
    for i in range(0,len(cells2)):
        for j in range(0,len(stats2_coords[i])):
            stats2_coords[i][j] = stats2_coords[i][j].tolist()
            stats2_coords[i][j] = [k for l in stats2_coords[i][j] for k in l]
            stats2_coords[i][j] = [int(stats2_coords[i][j][0]),int(stats2_coords[i][j][1])]
    
    #Loop through each set of points per ROI, check if the point (x and y) is found in the other file's ROI's, 
    #and return fraction of pts that overlap.
    overlap_list = np.zeros((len(cells1),len(cells2)))
    for i in range(0,len(cells1)):
        for k in range(0,len(cells2)):
            result = []
            for row in stats1_coords[i]:
                result.append(row in stats2_coords[k][:])
            overlap_list[i,k] = float(result.count(True)/len(result))
    return overlap_list

#Draws control points on an image. 
def controlpts():
    plt.waitforbuttonpress()
    while True: 
        points = []
        while len(points) < 6:
            points = np.asarray(plt.ginput(6,timeout=-1))
            break
            if len(points) < 5:
                print("Too few points - please choose 5.")
        break
    return points

#This takes in the baseline control points and currently analyzed control points, 
#And uses a lambda function to output the transformed coordinates of a point x. 
def transform(curr_pts,BL_pts):
    curr_pts = np.transpose(np.matrix(curr_pts))
    BL_pts = np.transpose(np.matrix(BL_pts))
    # vertically stack ones so that we can perform translations as well.
    curr_pts = np.vstack((curr_pts,np.ones((1,6))))
    BL_pts = np.vstack((BL_pts,np.ones((1,6))))
    # solve for the transformation matrix using the pseudo-inverse of our current point matrix. 
    #This ensures that it will still work regardless of whether any matrices are invertible or not.
    # use a lambda function to return the transformed coordinates of a transformed point x, 
    # using the transformation matrix from our BL and current points.
    t_matrix = BL_pts * np.linalg.pinv(curr_pts)
    return lambda x: (t_matrix*np.vstack((np.matrix(x).reshape(2,1),1)))[:2,:]

def adjusted_med_coordinates(stat_imgx,stat_imgy,statmed):
    min_x = min(stat_imgx)-10
    min_y = min(stat_imgy)-10
    max_x = min(stat_imgx)+10
    max_y = min(stat_imgy)+10
    difference_x = statmed[1]-(min_x)
    difference_y = statmed[0]-(min_y)
    good_coords = [difference_x,difference_y]
    return good_coords

def get_rois(path,rois_on):
    f = np.load(os.path.join(path,"F.npy"))
    Fneu = np.load(os.path.join(path, "Fneu.npy"))
    iscell = np.load(os.path.join(path,"iscell.npy"))
    spikes = np.load(os.path.join(path,"spks.npy"))
    stat = np.load(os.path.join(path,"stat.npy"),allow_pickle=True)
    ops = np.load(os.path.join(path,"ops.npy"),allow_pickle=True).item()
    
    #Get list of 'valid' cells, i.e. ROIs that are cells.
    cellValid = iscell[:,0]
    np_ops = np.array(ops['meanImg'])
    len_np_ops = 512
    cellIDMap = np.zeros([len_np_ops,len_np_ops], dtype = int)
    validCellList = np.where(cellValid==1)
    validCellList = np.squeeze(validCellList)
    validCellList = list(validCellList)
    print(validCellList)
    #Create mask to be used to create the image - all ROIs including overlap
    #Overlap is included because otherwise contour drawing can get messed up.
    #This does not affect ROI detection as each ROI is only compared with those from other sessions.
    cellID=[]
    tmp = []
    # for iCell in range (0,len(validCellList)+1):
    for iCell in validCellList:
        print(iCell)
        tmp = iCell;
        cellID=cellID+[tmp]
        for n in range(0,np.size(stat[iCell]['ypix'])):
            cellIDMap[stat[tmp]['ypix'],stat[tmp]['xpix']] = iCell
    img = cellIDMap
        
    return img,validCellList,stat,ops


def match_suite2p_files():
    """
    Interactive pipeline to load or create a match file between suite2p sessions.
    Prompts for all inputs, steps through each file pair, plots control points,
    computes ROI overlaps, and finally saves the matches to a CSV.
    Returns:
        pd.DataFrame: the table of matched ROIs.
    """
    # initialize
    match_df = pd.DataFrame()
    
    # load or start new
    load_match = input("Load previous match file? Type 'y' or 'n': ").strip().lower()
    if load_match == 'y':
        directory      = input("Directory of .mat file: ").strip()
        os.chdir(directory)
        match_file     = input("Match CSV filename to load: ").strip()
        match_df       = pd.read_csv(match_file, index_col=False)
        First_path     = input("Baseline suite2p file path: ").strip()
        n_files        = int(input("How many more files to analyze? ").strip())
        rois_on        = input("Show ROI numbers on plots? (True/False): ").strip() == 'True'
        exclusion_thr  = float(input("ROI inclusion threshold: ").strip())
        save_directory = input("Directory to save final match CSV: ").strip()
    else:
        First_path     = input("Baseline suite2p file to analyze: ").strip()
        BL_name        = input("Name for baseline session: ").strip()
        n_files        = int(input("How many files to analyze? ").strip())
        rois_on        = input("Show ROI numbers on plots? (True/False): ").strip() == 'True'
        exclusion_thr  = float(input("ROI inclusion threshold: ").strip())
        save_directory = input("Directory to save final match CSV: ").strip()
    
    # loop through new sessions
    for _ in range(n_files):
        matplotlib.use('TkAgg')
        Next_path      = input("Next suite2p file to analyze: ").strip()
        session_name   = input("Name for this session: ").strip()
        
        # draw images & get stats
        img1, cells1, stat1, ops1 = draw_image(First_path, rois_on)
        img2, cells2, stat2, ops2 = draw_image(Next_path, rois_on)
        
        # if new, seed the dataframe
        if load_match != 'y':
            match_df[BL_name] = cells1
        
        # plot baseline/control points
        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title("Draw 6 control points on baseline")
        ax1.imshow(ops1['meanImg'], cmap='gray')
        ax2 = fig.add_subplot(1,2,2)
        ax2.imshow(ops2['meanImg'], cmap='gray')
        BL_pts = controlpts()
        plt.close(fig)
        
        # annotate & draw points on current
        fig = plt.figure(figsize=(10,7))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(ops1['meanImg'], cmap='gray')
        for i,pt in enumerate(BL_pts): ax1.annotate(str(i+1), xy=(pt[0],pt[1]), color='red')
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title("Draw 6 control points on current")
        ax2.imshow(ops2['meanImg'], cmap='gray')
        curr_pts = controlpts()
        plt.close(fig)
        
        # compute transform & overlaps
        transform_fn = transform(curr_pts, BL_pts)
        overlaps     = roi_intersect(stat1, stat2, cells1, cells2, transform_fn)
        
        # pick best matches above threshold
        best_idx   = []
        matched    = []
        for ov_row in overlaps:
            max_ov = np.max(ov_row)
            if max_ov > exclusion_thr:
                idx = np.argmax(ov_row)
                best_idx.append(idx)
                matched.append(cells2[idx])
            else:
                best_idx.append(None)
                matched.append(None)
        
        # record in dataframe
        match_df[session_name] = matched
        
        # detailed plotting of only-overlaps (optional, can remove if too slow)
        valid_pairs = [(c1, m) for c1, m in zip(cells1, matched) if m is not None]
        plt.figure(figsize=(10, len(valid_pairs)*2))
        for i, (c1, c2) in enumerate(valid_pairs):
            # baseline ROI
            ax = plt.subplot2grid((len(valid_pairs),2),(i,0))
            ax.set_title(f"Baseline ROI {c1}")
            yps = stat1[c1]['ypix']; xps = stat1[c1]['xpix']
            ax.imshow(ops1['meanImg'][min(yps)-10:max(yps)+10, min(xps)-10:max(xps)+10], cmap='gray')
            xc, yc = tc.adjusted_med_coordinates(xps,yps,stat1[c1]['med'])
            ax.annotate(str(c1), xy=(xc,yc))
            # current ROI
            ax2 = plt.subplot2grid((len(valid_pairs),2),(i,1))
            ax2.set_title(f"Current ROI {c2}")
            yps2 = stat2[c2]['ypix']; xps2 = stat2[c2]['xpix']
            ax2.imshow(ops2['meanImg'][min(yps2)-10:max(yps2)+10, min(xps2)-10:max(xps2)+10], cmap='gray')
            xc2, yc2 = adjusted_med_coordinates(xps2,yps2,stat2[c2]['med'])
            ax2.annotate(str(c2), xy=(xc2,yc2))
        plt.show()
        
        # allow manual exclusion of bad matches
        bad = input("List any mismatched ROI numbers (space-sep), or press Enter: ")
        if bad:
            for num in bad.split():
                # remove any matched that equals this number
                for col in match_df.columns:
                    match_df.loc[match_df[col] == int(num), col] = None
        
    # save results
    os.chdir(save_directory)
    out_name = input("Filename to save matches as (with .csv): ").strip()
    match_df.to_csv(out_name, index=False)
    print(f"Saved match file to {os.path.join(save_directory, out_name)}")
    
    return match_df


if __name__ == '__main__':
    df = match_suite2p_files()

