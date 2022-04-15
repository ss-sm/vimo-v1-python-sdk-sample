from datetime import datetime


import cv2
from PIL import Image
import numpy as np
import yaml


from vimo_segmentation import vimo_segmentation


# 設定の読み込み
with open('settings.yaml', 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)


if __name__ == "__main__":
    # --------- ここから初期化 ---------
    engine = vimo_segmentation.engine()
    code = engine.Init(settings['SEGMENTATION_MODEL'], False, 0)
    request = vimo_segmentation.request()
    response = vimo_segmentation.response()
    
    # --------- 推論の実行 ---------
    request.image = cv2.imread(settings['SEGMENTATION_TEST_IMAGE'])
    request.thresholds = [3, 0] 
    code = engine.run(request, response)

    # --------- 推論結果の整形・出力 ---------
    img_gray = response.mask * 100
    img = np.zeros_like(request.image)
    img[:, :, 0] = img_gray
    result_img = request.image + img
    pil_img = Image.fromarray(result_img)
    filename = datetime.strftime(
        datetime.now(), 
        'files/inferred/test_result_%Y%m%d_%H%M%S.jpg'
    )
    pil_img.save(filename)
    print('finished!')