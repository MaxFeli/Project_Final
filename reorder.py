mdist_ul = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_dl = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_ur = [img.shape[0]**2 + img.shape[1]**2, 0]
    mdist_dr = [img.shape[0]**2 + img.shape[1]**2, 0]
    for item in range(len(points)):
        dist_ul = (points[item][0])**2 + (points[item][1])**2
        dist_dl = (points[item][0])**2 + (points[item][1]-img.shape[0])**2
        dist_ur = (points[item][0]-img.shape[1])**2 + (points[item][1])**2
        dist_dr = (points[item][0]-img.shape[1])**2 + (points[item][1]-img.shape[0])**2
        if dist_ul < mdist_ul[0]:
            mdist_ul[0] = dist_ul
            mdist_ul[1] = item
        if dist_dl < mdist_dl[0]:
            mdist_dl[0] = dist_dl
            mdist_dl[1] = item
        if dist_ur < mdist_ur[0]:
            mdist_ur[0] = dist_ur
            mdist_ur[1] = item
        if dist_dr < mdist_dr[0]:
            mdist_dr[0] = dist_dr
            mdist_dr[1] = item

    # define trapezoid and square, then warp mat
    srcSqr = np.array([points[mdist_ul[1]], points[mdist_ur[1]], points[mdist_dl[1]], points[mdist_dr[1]]])
    side = ((points[mdist_dl[1]][0]-points[mdist_dr[1]][0])**2+(points[mdist_dl[1]][1]-points[mdist_dr[1]][1])**2)**0.5
    dstSqr = np.array([points[mdist_dl[1]], points[mdist_dr[1]], points[mdist_dl[1]], points[mdist_dr[1]]])
    dstSqr[0][1] = dstSqr[0][1] - side
    dstSqr[1][1] = dstSqr[1][1] - side
    newSqr = np.array([0, 0])
    side_seg = side/8
    for item in range(len(points)):
        if item == 0:
            np.append(newSqr, dstSqr[0], axis=0)
            np.delete(newSqr, 0, axis=0)
        elif item % 9 != 0:
            np.append([newSqr[item-1][0]+side_seg, newSqr[item-1][1]], axis=0)
        else:
            np.append([newSqr[item-9][0], newSqr[item-9][1]+side_seg], axis=0)
    warp_mat = cv2.getAffineTransform(dstSqr, srcSqr)
    warp_dst = cv2.warpAffine(newSqr, warp_mat, (img.shape[0], img.shape[1]))
    # find nearest neighbour
    pointsSorted, index = {}, 0
    for i in range(len(points)):
        mdist = img.shape[0]**2 + img.shape[1]**2
        for j in range(len(points)):
            dist = (warp_dst[j][0]-points[i][0])**2+(warp_dst[j][1]-points[i][1])**2
            if dist < mdist:
                mdist = dist
                index = j
        pointsSorted[i] = points[j]
