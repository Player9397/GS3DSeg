from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

parent_dir = '/scratch/ashwin/gsplat/waldo_kitchen/images'
sam = sam_model_registry["vit_l"](checkpoint="/scratch/ashwin/gsplat/sam_vit_l_0b3195.pth")
sam = sam.to('cuda')
mask_generator = SamAutomaticMaskGenerator(sam)
annotation_path = os.path.split(parent_dir)[0] + '/Sam_annotations'
if os.path.exists(annotation_path):
    pass
else:
    os.makedirs(annotation_path)
for image_name in tqdm(os.listdir(parent_dir)):
    if os.path.exists(annotation_path+'/'+image_name+'.npy'):
        continue
    masks = mask_generator.generate(np.array(Image.open(parent_dir + '/'+image_name)))
    im_mask = np.ones([738, 994]) * 250
    for i, mask in enumerate(masks):
        # for k, val in mask.items():
        #     print(k, val)
        im_mask[np.array(mask['segmentation'], bool)] = i
    assert np.all(im_mask < 500)
    np.save(annotation_path+'/'+image_name+'.npy', im_mask)
    