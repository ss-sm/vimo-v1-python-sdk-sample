from datetime import datetime


import cv2
from PIL import Image, ImageDraw, ImageFont
import yaml


from vimo_detection import vimo_detection


# 設定の読み込み
with open('settings.yaml', 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)


if __name__ == "__main__":
    # --------- ここから初期化 ---------
    engine = vimo_detection.engine()
    code = engine.Init(settings['DETECTION_MODEL'], False, 0)
    request = vimo_detection.request()
    response = vimo_detection.response()
    # --------- 推論の実行 ---------
    request.image = cv2.imread(settings['DETECTION_TEST_IMAGE'])
    request.threshold = settings['THRESHOLD']
    start_time = datetime.now()
    engine.run(request, response)
    exe_time = datetime.now()-start_time
    print(f'execution time: {exe_time.seconds}.{exe_time.microseconds}')
    # --------- 推論結果の描画 ---------
    im = Image.open(settings['DETECTION_TEST_IMAGE'])
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", size=50)
    for box_info in response.box_list:
        draw.rectangle(
            (
                (box_info.xmin, box_info.ymax), 
                (box_info.xmax, box_info.ymin)
            ),
            outline=(0, 200, 200),
            width=8
        )
        draw.text(
            (box_info.xmin, box_info.ymax + 10),
            text=f'label: {box_info.label_id}, score: {box_info.score:.2f}',
            fill=(0, 200, 200),
            font=font
        )
    # --------- 推論結果のファイル出力 ---------    
    filename = datetime.strftime(
        datetime.now(), 
        './files/inferred/test_result_%Y%m%d_%H%M%S.jpg'
    )
    im.save(filename)
    print()
    print('finished')



