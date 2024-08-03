import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import torch
import supervision as sv


class InferlessPythonModel:
    def initialize(self):
        # nfs_volume = "/opt/tritonserver/vol"
        nfs_volume = os.getenv("NFS_VOLUME")
        if os.path.exists(nfs_volume + "/sam2_hiera_large.pt") == False :
            os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P {nfs_volume}")

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CHECKPOINT = f'{nfs_volume}/sam2_hiera_large.pt'
        CONFIG = "sam2_hiera_l.yaml"
        sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)
        self.mask_generator = SAM2ImagePredictor(sam2_model)

    def infer(self, inputs):
        image_base64_string = inputs["image_base64_string"]
        xmin = inputs["xmin"]
        ymin = inputs["ymin"]
        xmax = inputs["xmax"]
        ymax = inputs["ymax"]
        
        image_data = base64.b64decode(image_base64_string)
        image = Image.open(BytesIO(image_data))
        image_array = np.array(image)

        image_array_bgr = np.frombuffer(image_data, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array_bgr, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.mask_generator.set_image(image_rgb)
        coordinates = []
        
        for i in range(len(xmin)):
            coordinate = [
                int(xmin[i]),
                int(ymin[i]),
                int(xmax[i]),
                int(ymax[i]),
            ]
            coordinates.append(coordinate)

        masks, scores, _ = self.mask_generator.predict(
            box=np.array(coordinates),
            multimask_output=False,
        )
        
        if masks.shape[0] > 1:
            masks = np.squeeze(masks)
            
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks.astype(bool))
        segmented_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)
        image = Image.fromarray(segmented_image.astype('uint8'))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {"segmented_image_base64":img_str}
        

    def finalize(self):
        self.mask_generator = None
