from datetime import datetime
import pprint


import cv2
from PIL import Image, ImageDraw, ImageFont
import yaml


from vimo_ocr import vimo_ocr


# 設定の読み込み
with open('settings.yaml', 'r', encoding='utf-8') as f:
    settings = yaml.safe_load(f)


if __name__ == '__main__':
    # --------- ここから初期化 ---------
    engine = vimo_ocr.engine()
    engine.Init(settings['OCR_MODEL'], False, 0)
    request = vimo_ocr.request()
    response = vimo_ocr.response()
    request.image = cv2.imread(settings['OCR_TEST_IMAGE'])
    # --------- 推論の実行 ---------
    start_time = datetime.now()
    result = engine.run(request, response)
    exe_time = datetime.now()-start_time
    print(f'execution time: {exe_time.seconds}.{exe_time.microseconds}')
    # --------- 推論結果の描画 ---------
    im = Image.open(settings['OCR_TEST_IMAGE'])
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", size=20)
    for block in response.blocks:
        coordinates = block.polygon
        c = (
                (int(coordinates[0].x), int(coordinates[0].y)), 
                (int(coordinates[2].x), int(coordinates[2].y))
            )
        pprint.pprint(c)
        draw.rectangle(
            c,
            outline=(100, ),
            width=2
        )
        draw.text(
            (coordinates[0].x, coordinates[0].y - 25),
            text=block.text,
            fill=(0, ),
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