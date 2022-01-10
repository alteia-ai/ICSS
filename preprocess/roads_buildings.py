"""
Create a sub folder in Potsdam or Vaihingen folder containing a copy of images 
and a ground-truth containing only roads and buildings.
"""
import os
from glob import glob

import click
import cv2 as cv
from tqdm import tqdm


@click.command()
@click.option('-d', "--directory", help="Data directory", type=str, required=True)
def main(directory):
    imgs = glob(os.path.join(directory, "imgs/*"))
    gts = [i.replace("imgs", "gts") for i in imgs]
    new_folder = os.path.join(directory, "roads_buildings")
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        os.mkdir(new_folder+"/imgs")
        os.mkdir(new_folder+"/gts")
    for i, j in tqdm(zip(imgs, gts), total=len(imgs)):
        gt = cv.imread(j, 0)
        gt[gt==2] = 3
        gt[gt==1] = 2
        gt[gt==0] = 1
        gt[gt==4] = 0
        gt[gt==5] = 0
        gt[gt==3] = 0
        cv.imwrite(os.path.join(new_folder, j), gt)
    print("Conversion done.")

if __name__ == "__main__":
    main()
