"""生成一张用于跑通流水线的测试页：竖排日文对话气泡 + 旁白框。

真实漫画页效果更好，这张图只是保证没有素材时每一步都能单独验证。
运行：整合\.venv\Scripts\python.exe 整合\samples\make_sample_page.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_PATH = Path(__file__).resolve().parent / 'sample_page.png'
FONT_PATH = r'C:\Windows\Fonts\msgothic.ttc'
W, H = 900, 1300


def vertical_text(draw, cx, cy, text, font):
    step = font.size * 1.06
    y = cy - step * len(text) / 2 + step / 2
    for ch in text:
        draw.text((cx, y), ch, font=font, fill=(20, 20, 20), anchor='mm')
        y += step


def main():
    img = Image.new('RGB', (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([18, 18, W - 18, H - 18], outline=(25, 25, 25), width=3)

    font_b1 = ImageFont.truetype(FONT_PATH, 30)
    font_b2 = ImageFont.truetype(FONT_PATH, 30)
    font_narr = ImageFont.truetype(FONT_PATH, 26)

    draw.ellipse([150, 170, 470, 530], outline=(25, 25, 25), width=4)
    draw.polygon([(285, 515), (235, 610), (350, 520)], fill=(250, 250, 250),
                 outline=(25, 25, 25))
    vertical_text(draw, 310, 350, 'おはよう、いい天気だね', font_b1)

    draw.ellipse([520, 690, 800, 990], outline=(25, 25, 25), width=4)
    vertical_text(draw, 660, 840, 'ちょっと待ってよ！', font_b2)

    draw.rectangle([80, 1040, 420, 1230], outline=(25, 25, 25), width=3)
    vertical_text(draw, 250, 1135, '翌朝、駅前にて', font_narr)

    img.save(OUT_PATH)
    print(f'已生成 {OUT_PATH} ({W}x{H})')


if __name__ == '__main__':
    main()
