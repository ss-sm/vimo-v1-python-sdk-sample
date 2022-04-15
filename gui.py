import tkinter
import tkinter.filedialog
import tkinter.messagebox
import os
from datetime import datetime
import threading


import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk
import yaml


from vimo_detection import vimo_detection
# from vimo_ocr import vimo_ocr
# from vimo_segmentation import vimo_segmentation


# 設定の読み込み
class S:
    def __init__(self):
        if os.path.exists('settings.yaml'):
            self.read()
        else:
            self.settings = {
                "COLOR_B": 200,
                "COLOR_G": 200,
                "COLOR_R": 0,
                "FONT_SIZE": 100,
                "LINE_WIDTH": 15,
                "DETECTION_MODEL": "",
                "THRESHOLD": 0.7,
            }
            self.apply()
    def read(self):
        with open('settings.yaml', 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)
    def apply(self):
        with open('settings.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.settings, f, allow_unicode=True)

s = S()


# GUIアプリのセットアップ
app = tkinter.Tk()
app.resizable(0, 0)
app.geometry("800x600")
app.title(f"Sample App powered by ViMo SDK")
label = tkinter.Label(app, text='Nothing selected')
label.place(x=25, y=50)
select_button = tkinter.Button(app, text='Select target file')
select_button.place(x=25, y=15)
setting_button = tkinter.Button(app, text='Change settings')
setting_button.place(x=680, y=15)
detect_button = tkinter.Button(
    app,
    text='Execute',
    width=10
)
detect_button.place(x=350, y=560)
canvas = tkinter.Canvas(bg="black", width=750, height=460)
canvas.place(x=25, y=80)
raw_im = None
tk_im = None


# ViMo エンジンの初期化
detection_engine = vimo_detection.engine()
# segmentation_engine = vimo_segmentation.engine()



def select_file(event):
    """
    select button クリック時のイベント処理
    """
    global raw_im
    global tk_im
    try:
        fTyp = [("", "*")]
        file_name = tkinter.filedialog.askopenfilename(
            filetypes=fTyp, 
            initialdir="./files/uploaded"
        )
        if len(file_name) == 0:
            pass
        else:
            label['text'] = file_name
            raw_im = Image.open(file_name)
            w = raw_im.width
            h = raw_im.height
            raw_im = raw_im.resize(( int(w * (760/w)), int(h * (760/w)) ))
            tk_im = ImageTk.PhotoImage(image=raw_im)
            canvas.create_image(0, 0, anchor='nw', image=tk_im)
    except Exception as e:
        tkinter.messagebox.showinfo('失敗', e)
    return 'break'


def detect(event):
    """
    execute button クリック時のイベント処理
    """
    global raw_im
    global tk_im
    s.read()
    if not os.path.exists(s.settings['DETECTION_MODEL']):
        tkinter.messagebox.showinfo('失敗', '正しいAIモデルが選択されていません')
        return 'break'
    try:
        # --------- 推論エンジンの準備 ---------  
        detection_engine.Init(s.settings['DETECTION_MODEL'], False, 0)
        request = vimo_detection.request()
        response = vimo_detection.response()
        request.image = cv2.imread(label['text'])
        request.threshold = s.settings['THRESHOLD']
        # --------- 推論実行 ---------  
        start_time = datetime.now()
        detection_t = threading.Thread(
            target=detection_engine.run,
            args=(request, response)
        )
        detection_t.start()
        detection_t.join()
        exe_time = datetime.now()-start_time
        print(f'execution time: {exe_time.seconds}.{exe_time.microseconds}')
        # --------- 推論結果の描画 --------- 
        im = Image.open(label['text'])
        draw = ImageDraw.Draw(im)
        font = ImageFont.truetype("arial.ttf", size=s.settings['FONT_SIZE'])
        for box_info in response.box_list:
            draw.rectangle(
                (
                    (box_info.xmin, box_info.ymax), 
                    (box_info.xmax, box_info.ymin)
                ),
                outline=(s.settings['COLOR_R'], s.settings['COLOR_G'], s.settings['COLOR_B']),
                width=s.settings['LINE_WIDTH']
            )
            draw.text(
                (box_info.xmin, box_info.ymax + 10),
                text=f'label: {box_info.label_id}, score: {box_info.score:.2f}',
                fill=(s.settings['COLOR_R'], s.settings['COLOR_G'], s.settings['COLOR_B']),
                font=font
            )
        # --------- 推論結果のファイル出力 ---------    
        dir = os.path.abspath(os.path.dirname(__file__))
        filename = datetime.strftime(
            datetime.now(), 
            dir + '\\files\\inferred\\test_result_%Y%m%d_%H%M%S.jpg'
        )
        im.save(filename)
        # --------- 推論結果の画面出力 ---------  
        raw_im = Image.open(filename)
        w = raw_im.width
        h = raw_im.height
        raw_im = raw_im.resize(( int(w * (760/w)), int(h * (760/w)) ))
        tk_im = ImageTk.PhotoImage(image=raw_im)
        canvas.create_image(0, 0, anchor='nw', image=tk_im)
        tkinter.messagebox.showinfo('detection', 'finished')
    except Exception as e:
        tkinter.messagebox.showinfo('失敗', e)
    return 'break'


def change_settings(event):
    global app
    s.read()
    setting_modal = tkinter.Toplevel(app)
    setting_modal.resizable(0, 0)
    setting_modal.title("Settings")
    setting_modal.geometry("600x280")
    detection_label = tkinter.Label(setting_modal, text='Detection Settings')
    detection_label.place(x=25, y=25)
    model_label = tkinter.Label(setting_modal, text='Model:')
    model_label.place(x=25, y=60)
    model_text_var = tkinter.StringVar()
    model_text_var.set(s.settings['DETECTION_MODEL'])
    model_text = tkinter.Label(setting_modal, textvariable=model_text_var)
    model_text.place(x=100, y=60)
    select_model_button = tkinter.Button(setting_modal, text='Select Model')
    select_model_button.place(x=320, y=235)

    def select_model(event):
        fTyp = [("", "*.smartmore")]
        file_name = tkinter.filedialog.askopenfilename(
            filetypes=fTyp, 
            initialdir="./models"
        )
        if len(file_name) == 0:
            pass
        else:
            model_text_var.set(file_name)
        return 'break'

    select_model_button.bind('<ButtonPress>', select_model)
    threshold_label = tkinter.Label(setting_modal, text='Threshold:')
    threshold_label.place(x=25, y=95)
    threshold_text_box = tkinter.Entry(setting_modal)
    threshold_text_box.insert(0, s.settings['THRESHOLD'])
    threshold_text_box.place(x=100, y=95)
    color_label = tkinter.Label(setting_modal, text='Color:')
    color_r_label = tkinter.Label(setting_modal, text='R')
    color_g_label = tkinter.Label(setting_modal, text='G')
    color_b_label = tkinter.Label(setting_modal, text='B')
    color_r_text_box = tkinter.Entry(setting_modal)
    color_r_text_box.insert(0, s.settings['COLOR_R'])
    color_g_text_box = tkinter.Entry(setting_modal)
    color_g_text_box.insert(0, s.settings['COLOR_G'])
    color_b_text_box = tkinter.Entry(setting_modal)
    color_b_text_box.insert(0, s.settings['COLOR_B'])
    color_label.place(x=25, y=130)
    color_r_label.place(x=100, y=130)
    color_r_text_box.place(x=120, y=130)
    color_g_label.place(x=250, y=130)
    color_g_text_box.place(x=270, y=130)
    color_b_label.place(x=400, y=130)
    color_b_text_box.place(x=420, y=130)
    line_label = tkinter.Label(setting_modal, text='Line Width:')
    line_text_box = tkinter.Entry(setting_modal)
    line_text_box.insert(0, s.settings['LINE_WIDTH'])
    line_label.place(x=25, y=165)
    line_text_box.place(x=100, y=165)
    font_label = tkinter.Label(setting_modal, text='Font Size:')
    font_text_box = tkinter.Entry(setting_modal)
    font_text_box.insert(0, s.settings['FONT_SIZE'])
    font_label.place(x=25, y=200)
    font_text_box.place(x=100, y=200)

    def apply_settings(event):
        s.settings['DETECTION_MODEL'] = model_text['text']
        s.settings['COLOR_R'] = int(color_r_text_box.get())
        s.settings['COLOR_G'] = int(color_g_text_box.get())
        s.settings['COLOR_B'] = int(color_b_text_box.get())
        s.settings['THRESHOLD'] = float(threshold_text_box.get())
        s.settings['FONT_SIZE'] = int(font_text_box.get())
        s.settings['LINE_WIDTH'] = int(line_text_box.get())
        s.apply()
        tkinter.messagebox.showinfo('apply settings', f'All settings are applied!')
        return 'break'

    apply_button = tkinter.Button(setting_modal, text='Apply All')
    apply_button.place(x=240, y = 235)
    apply_button.bind('<ButtonPress>', apply_settings)

    setting_modal.grab_set()
    setting_modal.focus_set()
    app.wait_window(setting_modal)
    return 'break'


if __name__ == '__main__':
    select_button.bind('<ButtonPress>', select_file)
    detect_button.bind('<ButtonPress>', detect)
    setting_button.bind('<ButtonPress>', change_settings)
    app.mainloop()