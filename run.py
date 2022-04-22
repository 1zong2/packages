import os
from tkinter import E
import cv2
import sys
sys.path.append("../")
import glob
from PIL import Image
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    # ### face alignment
    # from face_alignment import do_align
    # from facenet_pytorch import MTCNN 
    # image_paths = glob.glob("samples/*.*")
    # mtcnn = MTCNN()
    # for image_path in image_paths:
    #     image_name = os.path.split(image_path)[1][:-4]
    #     image = Image.open(image_path)
    #     aligned_iamge = do_align(image, output_size=256)[0]
    #     cv2.imwrite(f"outputs/{image_name}.jpg", aligned_iamge[:, :, ::-1])

    # ### ID extraction
    from arcface import get_id
    # from curricularface import get_id
    source_paths = sorted(glob.glob("outputs/j*.*"))
    target_paths = sorted(glob.glob("outputs/wb*.*"))
    dummy0 = np.zeros((256,256,3))
    dummy1 = np.ones((256,256,3))*127
    dummy2 = np.ones((256,256,3))*191
    cs = nn.CosineSimilarity()

    source_image_list = []
    source_id_list = []
    target_image_list = []
    target_id_list = []
    score_list = []

    for source_path in source_paths:
        source_image = Image.open(source_path).resize((256,256))
        source_id_vector = get_id(source_image)
        source_image_list.append(np.array(source_image))
        source_id_list.append(source_id_vector)

    for target_path in target_paths:
        # image_name = os.path.split(image_path)[1][:-4]

        target_image = Image.open(target_path).resize((256,256))
        target_id_vector = get_id(target_image)
        target_image_list.append(np.array(target_image))
        target_id_list.append(target_id_vector)

    score_list = []
    for source_id_vector in source_id_list:
        row = []
        for target_id_vector in target_id_list:
            score = (1-cs(source_id_vector, target_id_vector))
            row.append(np.round(float(score.detach().cpu().numpy()),2))

        score_list.append(row)
    print(score_list)

    grid = []
    grid.append([dummy0]+source_image_list)
    for i in range(len(target_paths)):
        row = [target_image_list[i]]
        for j in range(len(source_paths)):
            if j%2 == i%2:
                row.append(dummy1)
            else: 
                row.append(dummy2)
        grid.append(row)
    grid = np.concatenate(grid, axis=1)
    grid = np.concatenate(grid, axis=1)

    cv2.imwrite("grid.jpg", np.array(grid)[:, :, ::-1])