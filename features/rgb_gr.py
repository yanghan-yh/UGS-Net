import os
import cv2

# data_path = 'test/'
# out_path = 'res/'
# filelist = os.listdir(data_path)

def trans(img):
    ht = img.shape[0]
    wd = img.shape[1]
    for i in range(0,ht):
        for j in range(0,wd):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if B <= 51:
                img[i][j][0] = 255
            elif B <= 102:
                img[i][j][0] = 255 - (B-51)*5
            elif B <= 153:
                img[i][j][0] = 0
            else : img[i][j][0] = 0

            if G <= 51:
                img[i][j][1] = G*5
            elif G <= 102:
                img[i][j][1] = 255
            elif G <= 153:
                img[i][j][1] = 255
            elif G <= 204:
                img[i][j][1] = 255 - int(128.0*(G-153.0)/51.0 + 0.5)
            else : img[i][j][1] = 127 - int(127.0*(G-204.0)/51.0 + 0.5)

            if R <= 51:
                img[i][j][2] = 0
            elif R <= 102:
                img[i][j][2] = 0
            elif R <= 153:
                img[i][j][2] = (R-102)*5
            elif G <= 204:
                img[i][j][2] = 255
            else : img[i][j][2] = 255
    # cv2.imwrite(os.path.join(out_path,file_name),img)
    return img


# for onefile in filelist:
#     img = cv2.imread(data_path + onefile)
#     trans(onefile, img)