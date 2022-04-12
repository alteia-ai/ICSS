# Presentation
This repository contains the code of our [paper](https://arxiv.org/abs/2201.01029): Weakly-supervised continual learning for class-incremental segmentation. 


# To use
## To install
```Shell
    conda create -n ICSS python gdal=2.4.4 shapely rtree -c 'conda-forge'
    conda activate ICSS
    pip install -r requirements.txt
```

## Prepare data
The training datasets should be stored in a folder *MyDataset* organized as follows:
 - a folder named `imgs` containing the RGB images.
 - a folder named `gts` containing the ground-truths.

:warning: Ground-truth files must have the same names than their associated image.

#### Example for [ISPRS Potsdam dataset](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html).

```Shell
cd <PotsdamDataset>
sudo apt install rename
cd gts; rename 's/_label//' *; cd ../imgs; rename 's/_RGB//' *
```
The ground-truth maps have to be one-hot encoded (i.e. not in a RGB format):
```Shell
cd ICSS
python preprocess/format_gt.py -n 6 -d <PathToMyDataset>/gts
```
To create a sub-folder with only roads and buildings:
```Shell
python preprocess/roads_buildings.py -d <PathToMyDataset>
ln -s <PathToMyDataset>/imgs <PathToMyDataset>/buildings_cars/imgs
```

## To train:
```Shell
python -m src.train -d /data/gaston/Potsdam/roads_buildings -c configs/some_config.yml
```

## To add a semantic segmentation class: 
```Shell
python -m src.increment_class -d /data/gaston/Potsdam/roads_buildings -c configs/some_config.yml -p data/models/LinkNet34_Potsdam__template.pt
```

## To infer
```Shell
python -m src.infer -c configs/some_config.yml -o ~/preds -i ~/data/Potsdam/imgs/top_potsdam_7_12.tif  -p data/models/LinkNet34_roads_buildings_template.pt
```
 
# Licence

Code is released under the MIT license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

See [LICENSE](./LICENSE) for more details.

# Acknowledgements

This work has been jointly conducted at [Alteia](https://alteia.com/)  and [ONERA-DTIS](https://www.onera.fr/en/dtis).
