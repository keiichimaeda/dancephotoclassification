import os
import numpy as np
import face_recognition
from PIL import Image, ImageDraw, ExifTags

# "images"ディレクトリ内のすべてのJPEGファイルを取得
image_dir = "images"
output_dir = "outputs"
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg"))]
print(image_files)

for image_file in image_files:
    # 画像を読み込む
    load_image = face_recognition.load_image_file(os.path.join(image_dir, image_file))

    # 画像ファイルの情報を出力
    print(f"File name: {image_file}")
    print(f"File size: {os.path.getsize(os.path.join(image_dir, image_file))} bytes")
    print(f"File path: {os.path.join(image_dir, image_file)}")

    # デバッグ情報を出力
    print(f"Processing file: {image_file}")

    height, width, _ = load_image.shape

    # メタ情報から縦横を判断
    pil_image = Image.open(os.path.join(image_dir, image_file))
    exif = pil_image._getexif()

    rotation = 0
    is_portrait = False
    if exif:
        for tag, value in exif.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == 'Orientation':
                print(value)
                if value in [8]:  #　8: 270度回転
                    is_portrait = True
                    rotation = 1
                    print("Orientation: 270度")
                if value in [6]:  # 6: 90度回転
                    is_portrait = True
                    rotation = 3
                    print("Orientation: 90度")
                else:
                    print("Orientation: Landscape")
                break
    else:
        # メタ情報がない場合は画像のサイズから判断
        if height > width:
            is_portrait = True
            print("Orientation: Portrait")
        else:
            print("Orientation: Landscape")

    # 縦画像だったら90度回転させる
    if is_portrait:
        load_image = np.rot90(load_image, rotation)
        height, width = width, height  # 回転後のサイズを更新
        print(f"Rotated size: {width}x{height}")

    # 認識させたい画像から顔検出する
    face_locations = face_recognition.face_locations(load_image)
    print(f"Detected {len(face_locations)} face(s)")

    pil_image = Image.fromarray(load_image)
    draw = ImageDraw.Draw(pil_image)

    # 検出した顔分ループする
    for (top, right, bottom, left) in face_locations:
        # 顔の周りに四角を描画する
        draw.rectangle(((left, top), (right, bottom)),
                       outline=(255, 0, 0), width=2)

    del draw

    # EXIF情報を再設定して結果の画像を保存する
    pil_image.save(os.path.join(output_dir, image_file))
    print(f"Saved result to {os.path.join(output_dir, image_file)}")