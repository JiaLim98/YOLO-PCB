import torch
import pandas as pd
import csv

def size(xywh, classes, img_inspected, size_class, defect_list, inc, tol=5, last=8):
    # param = last seen size (maximum of the batch)
    # to account for 1D tensor
    num = torch.numel(xywh)
    row = int(num/4)
    a = classes.reshape(row,1)
    b = xywh.reshape(row,4)
    pred = torch.cat([a,b],1) 

    # convert sizes to their respective ones
    pred_1 = torch.tensor([]).cuda()
    for i in range(len(pred)):
        if pred[i,0] == 0:          # missing hole diameter = (w+h)/2
           sz_tmp = (pred[i,3] + pred[i,4])/2
        else:
           rel = pred[i,3]/pred[i,4] # w/h ratio
           if rel < 0.5:
              sz_tmp = pred[i,4]    # w/h << 1, choose h
           elif rel > 1.5:
              sz_tmp = pred[i,3]    # w/h >> 1, choose w
           else:
              sz_tmp = torch.sqrt(pred[i,3]**2+pred[i,3]**2)  # hypotenuse

        pred_1 = torch.cat([pred_1,pred[i,0].unsqueeze(0),sz_tmp.unsqueeze(0)])
    pred_1 = pred_1.reshape(int(torch.numel(pred_1)/2),2)
    param_old = size_class[:,1]
    param_new = torch.tensor([]).cuda()
    for j in range(0,7):
        det_tmp = pred_1[:,0]==j
        max_tmp = max(pred_1[:,1][det_tmp]).unsqueeze(0) if torch.any(det_tmp) else torch.tensor([0]).cuda()
        param_new = torch.cat([param_new,max_tmp])
    up = param_new > param_old
    up = up * 1

    # create last seen tensor
    size_ls = torch.tensor([]).cuda()
    for m in range(0,7):
        nm = pred_1[:,0]==m
        if torch.any(nm):
           ls = max(img_inspected[nm]).unsqueeze(0)
        else:
           ls = size_class[m,3].unsqueeze(0)
        size_ls = torch.cat([size_ls,ls])

    for n in range(0,7):
        if size_ls[n]-size_class[n,3] < last: # last seen range lower than allowable
           size_class[n,2] = size_class[n,2] + up[n]
           size_class[n,1] = param_new[n] if up[n] == 1 else param_old[n]
           if size_class[n,2] >= tol and n != 0 and up[n] !=0:
              print('Defect: %s is increasingly bigger.' % defect_list[n-1])
              # inc = torch.cat([inc,torch.tensor([img_inspected[0],n]).cuda()])
        else:
           size_class[n,2] = up[n]
           size_class[n,1] = torch.tensor([0]) # param_old[n]
    # size_class[:,1] = param_new
    size_class[:,3] = size_ls[:]

    return size_class

def rep(img,xywh,occ_point,sz=100,tol=1, rep=3, last=5):
    # OCC_POINT MUST HAVE SOMETHING, PUT IF STATEMENT OUTSIDE OF THIS FUNCTION
    # img is created by the algo itself, string filename doesnt permit 'last seen' purposes
    # different image will be assigned different int filename outside
    
    if img is None:       # zero defects detected (may not be None afterwards)
       return occ_point

    if len(occ_point) >= sz:
       ex = len(occ_point)-sz
       occ_point = occ_point[ex-1:,:]

    # to account for 1D tensor (one defect detected)
    num, num_occ, col = torch.numel(xywh),torch.numel(occ_point),4
    row, row_occ = int(num/col), int(num_occ/col)
    xywh = xywh.reshape(row,col)
    filename = img.reshape(row,1)
    occ_point = occ_point.reshape(row_occ,col)
    x = (xywh[:,0]).reshape(row,1)  # new mid-point x
    y = (xywh[:,1]).reshape(row,1)  # new mid-point y
    xy = torch.cat([x,y],1)

    pot = torch.tensor([]).cuda()  # potential clusters
    npot = torch.tensor([]).cuda()  # non-potential clusters
    for i in occ_point[:,0:2]:
        pot_tmp = torch.tensor([]).cuda()
        npot_tmp = torch.tensor([]).cuda()
        for j in range(len(xy)):
            dist = torch.sqrt((i[0]-xy[j][0])**2+(i[1]-xy[j][1])**2) # euclidean distance
            if (dist<=tol).item() is True: 
               pot_tmp = torch.cat([pot_tmp,i,filename[j],xy[j],filename[j]],0)
            else: # if more than, set number of occurrences to zero in occ_point
               npot_tmp = torch.cat([npot_tmp,xy[j],torch.tensor([-1]).cuda()],0)
        pot = torch.cat([pot,pot_tmp],0)
        npot = torch.cat([npot,npot_tmp],0)

    npot_list = npot.reshape(int(torch.numel(npot)/3),3)
    pot_list = pot.reshape(int(torch.numel(pot)/3),3)
    ind, ind_cnts = torch.unique(pot_list[:,0:2], dim=0, return_counts=True)  # individual points and their occurrences

    # create ind_file tensor for 'last seen'
    ind_ls = torch.tensor([]).cuda()
    for k in ind:
        nm = torch.all(pot_list[:,0:2]==k,dim=1)
        ls = min(pot_list[:,2][nm]).unsqueeze(0)
        ind_ls = torch.cat([ind_ls,ls],dim=0)

    # merging non-potential clusters
    ind = torch.cat([ind,npot_list[:,0:2]],0)
    ind_cnts = torch.cat([ind_cnts,torch.zeros(npot_list.shape[0]).cuda()],0)
    ind_ls = torch.cat([ind_ls,npot_list[:,2]],0)

    for m in range(len(ind)):
        pos = torch.all(occ_point[:,0:2]==ind[m],dim=1).nonzero(as_tuple=True)
        if len(pos[0]) != 0: # check if point existed in cache
           if ind_ls[m]-occ_point[pos,2] >= last:
              occ_point[pos,2] = ind_ls[m].double()
              occ_point[pos,3] = ind_cnts[m].double()
           else:
              occ_point[pos,3] = occ_point[pos,3]+ind_cnts[m]
              if occ_point[pos,3] >= rep:
                 print('Cluster Appearing at x = %.3f , y = %.3f' % (occ_point[pos,0], occ_point[pos,1]))
        else: # concatenate if point did not exist in cache
           occ_point_tmp = torch.cat([ind[m],ind_ls[m].unsqueeze(0),ind_cnts[m].unsqueeze(0)])
           occ_point_tmp = occ_point_tmp.reshape(1,4)
           occ_point = torch.cat([occ_point,occ_point_tmp],dim=0)

    return occ_point

def cnt(classes,img_inspected,cnt_class,defect_list,inc,tol=5,last=6):
    # no_img_inspected = how many images were inspected in this batch
    # classes should be a 1D tensor
    # param (batch by batch)= number of defects/number of inspected images
    # cnt_class = [defect, detected so far, total number of inspected images, num increase within tol, last seen increase]
    no_img_inspected = torch.max(img_inspected)-torch.min(img_inspected)+1        # number of images inspected in this batch
    class_cnt = torch.arange(7).cuda()                                            # class count -> 0123456
    classes_tmp = torch.cat([class_cnt,classes])                                  # class temporary for individual counts
    ind, ind_cnts = torch.unique(classes_tmp, return_counts=True)
    ind_cnts -= 1                                                                 # individual counts
    param_old = cnt_class[:,1]/cnt_class[:,2]                                     # old parameter
    param_new = (cnt_class[:,1]+ind_cnts)/(cnt_class[:,2]+no_img_inspected)       # new parameter
    up = param_new > param_old
    up = up * 1                                                                   # increase count

    cnt_ls = torch.tensor([]).cuda()                                              # create last seen tensor
    for j in class_cnt:
        nm = classes==j                                                           # determine if class exists
        if torch.any(nm):
           ls = max(img_inspected[nm]).unsqueeze(0)                               # if yes, maximum img number as last seen
        else:
           ls = cnt_class[j,4].unsqueeze(0)                                       # if no, remain as previous last seen
        cnt_ls = torch.cat([cnt_ls,ls])

        if cnt_ls[j]-cnt_class[j,4] < last:                                       # last seen range lower than allowable
           sml = cnt_class[j,3]                                                   # similarity check with previous
           cnt_class[j,3] = cnt_class[j,3] + up[j]                                # add
           if cnt_class[j,3] >= tol and j != 0 and up[j] != 0:                      # increased higher than allowable
              print('Defect: %s is increasing.' % defect_list[i-1])
              # inc = torch.cat([inc,torch.tensor([img_inspected[0],j]).cuda()])    # for visualization
        else:                                                                     # last seen range higher than allowable
           cnt_class[j,3] = up[j]                                                 # resets zero/one if not seen
    
    cnt_class[:,1] = cnt_class[:,1] + ind_cnts                                    # update total defects detected
    cnt_class[:,2] = cnt_class[:,2] + no_img_inspected                            # update total images inspected
    cnt_class[:,4] = cnt_ls[:]                                                    # update all last seen
    
    return cnt_class