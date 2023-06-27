import os.path
import tempfile

import cv2
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Image, HRFlowable, Table

# 注册字体
pdfmetrics.registerFont(TTFont('SimSun', os.path.abspath('report/fonts/SimSun.ttc')))
pdfmetrics.registerFont(TTFont('SimHei', os.path.abspath('report/fonts/SimHei.ttf')))


def convert_to_Image(image, scale):

    height = image.shape[0]
    width = image.shape[1]

    # resize_image = cv2.resize(image, (int(width/scale), int(height/scale)), interpolation=cv2.INTER_AREA)
    temp_file = tempfile.NamedTemporaryFile()
    temp_file_path = temp_file.name + '.png'
    # cv2.imwrite(temp_file_path, resize_image)
    cv2.imwrite(temp_file_path, image)

    image_reportlab = Image(temp_file_path)
    image_reportlab.drawWidth = width/scale
    image_reportlab.drawHeight = height/scale

    temp_file.close()

    return image_reportlab


class Graphics:
    # 绘制标题
    @staticmethod
    def draw_title_1(title: str):
        style = getSampleStyleSheet()
        ct = style['Heading1']
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 20  # 字体大小
        ct.leading = 20  # 行间距
        ct.textColor = colors.black  # 字体颜色
        ct.alignment = 1  # 居中
        ct.bold = True
        return Paragraph(title, ct)

    # 绘制小标题
    @staticmethod
    def draw_title_2(title: str):
        style = getSampleStyleSheet()
        ct = style['Heading2']
        ct.fontName = 'SimSun'  # 字体名
        ct.fontSize = 18  # 字体大小
        ct.leading = 30  # 行间距
        ct.textColor = colors.black  # 字体颜色
        ct.alignment = 1  # 居中
        return Paragraph(title, ct)

    @staticmethod
    def draw_title_3(title: str):
        style = getSampleStyleSheet()
        ct = style['Normal']
        ct.fontName = 'SimHei'  # 字体名
        ct.fontSize = 12  # 字体大小
        ct.leading = 25  # 行间距
        ct.textColor = colors.black  # 字体颜色
        return Paragraph(title, ct)

    @staticmethod
    def draw_text(text: str, space):
        style = getSampleStyleSheet()
        normal_style = style['Normal']
        ct = ParagraphStyle(name='custom_style', parent=normal_style, leftIndent=space)
        ct.fontName = 'SimSun'
        ct.fontSize = 12
        ct.wordWrap = 'CJK'  # 自动换行
        ct.alignment = 0  # 左对齐
        ct.leading = 22
        return Paragraph(text, ct)

    # 绘制图片
    @staticmethod
    def draw_img(image):
        return convert_to_Image(image, 1)

    @staticmethod
    def draw_img_row(image_1, image_2, width):
        image1 = convert_to_Image(image_1, 5)
        image2 = convert_to_Image(image_2, 5)

        data = [[image1, image2]]

        table_style = [
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 居中对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 垂直居中对齐
            ('GRID', (0, 0), (-1, -1), 1, colors.transparent),  # 设置网格线为透明
        ]
        return Table(data, style=table_style, colWidths=[width / 2] * 2)

    @staticmethod
    def draw_table(data, width):
        col_num = len(data[0])

        table_style = [
            ('GRID', (0, 0), (-1, -1), 1, colors.transparent),  # 设置网格线为透明
            ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('LEADING', (0, 0), (-1, -1), 25),
        ]
        return Table(data, style=table_style, colWidths=[width / col_num] * col_num)

    @staticmethod
    def draw_signature(data, width):
        col_num = len(data[0])

        table_style = [
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # 居中对齐
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # 垂直居中对齐
            ('GRID', (0, 0), (-1, -1), 1, colors.transparent),  # 设置网格线为透明
            ('FONTNAME', (0, 0), (-1, -1), 'SimSun'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
        ]
        return Table(data, style=table_style, colWidths=[width / col_num] * col_num)

    @staticmethod
    def draw_line():
        line = HRFlowable(width="100%", thickness=1, lineCap="round", color=colors.black)
        return line
