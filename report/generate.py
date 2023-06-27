import os
import random
from datetime import datetime
from glob import glob

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import SimpleDocTemplate, Spacer

from algorithm import ROISizeAlgorithm, CalcificationClassificationAlgorithm, SegmentationAlgorithm, PositionAlgorithm, \
    NoduleInnerMassAlgorithm, InnerMassClassificationAlgorithm, EdgeAlgorithm, NoduleCalcificationAlgorithm, \
    HaloAlgorithm
from report.graphics import Graphics, convert_to_Image
from utils.misc import boolean_to_zh_cn

# 注册字体
pdfmetrics.registerFont(TTFont('SimSun', os.path.abspath('report/fonts/SimSun.ttc')))
pdfmetrics.registerFont(TTFont('SimHei', os.path.abspath('report/fonts/SimHei.ttf')))

signatures = glob(os.path.abspath('report/signatures/*.png'))


class Report:

    def __init__(self, image, seg, props, page_size=A4, margin=30):
        self.page_width = page_size[0]
        self.page_height = page_size[1]
        self.margin = margin
        self.image = image.copy()

        self.desc = PositionAlgorithm.get_image_and_description(image, seg, props)[1]
        self.exist = SegmentationAlgorithm.get_image_and_description(image, seg, props)[2]
        self.calcification = CalcificationClassificationAlgorithm.get_image_and_description(image, seg, props)[0]
        if not self.exist:
            return
        # 结节大小、纵横比
        ud, lr, fb = ROISizeAlgorithm.get_image_and_description(image, seg, props)[2]
        if ud == 0:
            self.size = f"左右径{lr:.2f}mm，前后径{fb:.2f}mm"
            if fb / lr > 1:
                self.radio = "纵横比>1，"
            else:
                self.radio = "纵横比<1，"
        else:
            self.size = f"上下径{ud:.2f}mm，前后径为{fb:.2f}mm"
            self.radio = ""
        # 内质囊实性
        self.composition = NoduleInnerMassAlgorithm.get_image_and_description(image, seg, props)[2]
        # 边缘
        self.clear, self.smooth = EdgeAlgorithm.get_image_and_description(image, seg, props)[2]
        # 钙化
        self.Cal = NoduleCalcificationAlgorithm.get_image_and_description(image, seg, props)[1]
        # 回声
        self.echo = InnerMassClassificationAlgorithm.get_image_and_description(image, seg, props)[2]
        if self.echo == '无法判断回声':
            self.echo = ''
        # 微钙化
        self.microCal = CalcificationClassificationAlgorithm.get_image_and_description(image, seg, props)[2]
        # 声晕
        self.halo = HaloAlgorithm.get_image_and_description(image, seg, props)[1]
        # CTIRAD
        self.ctirad = ""

    def render_footer(self, c: Canvas, date=datetime.now()):
        """绘制页脚"""
        c.setStrokeColor(colors.dimgrey)
        c.line(30, self.page_height - 780, 570, self.page_height - 780)
        c.setFont('SimSun', 10)
        c.setFillColor(colors.black)
        c.drawString(30, self.page_height - 798, f"本报告由自动化代码生成，仅供答辩参考，不作为法律依据")
        c.drawString(30, self.page_height - 815, f"生成日期：{date.strftime('%Y.%m.%d %H:%M:%S ')}")

    # 渲染页眉
    def render_header(self, c: Canvas):
        """绘制页眉"""
        logo = os.path.abspath('report/logo.png')
        c.drawImage(logo, 30, self.page_height - 25 - 20, width=175, height=25, mask='auto')

    def my_first_page(self, c: Canvas, doc):
        c.saveState()
        self.render_header(c)
        self.render_footer(c)
        c.restoreState()

    def my_later_pages(self, c: Canvas, doc):
        c.saveState()
        self.render_header(c)
        self.render_footer(c)
        c.restoreState()

    def generate_content(self) -> str:
        if self.exist:
            return f"甲状腺内见{self.size}的{self.echo}结节，{self.radio}内部结构{self.composition}，" \
                   f"边界{boolean_to_zh_cn(self.clear)}清晰，形态{boolean_to_zh_cn(self.smooth)}光滑，{self.halo}，" \
                   f"内{self.Cal}。"
        else:
            return "甲状腺部分无结节。"

    def cal_CTIRAD(self) -> int:
        if self.exist:
            level = 0
            criterion_name = ["垂直位，", "实性，", "极低回声，", "点状强回声，", "边缘模糊/不规则"]
            criterions = [self.radio == "纵横比>1，", self.composition == '实性', self.echo == '极低回声', self.microCal,
                          not (self.clear and self.smooth)]
            for idx, criterion in enumerate(criterions):
                level += int(criterion)
                if criterion:
                    self.ctirad += criterion_name[idx]
            return level
        else:
            return 0

    def create_pdf(self, pdf_name):
        content = list()
        page_width = self.page_width - 2 * self.margin

        content.append(Graphics.draw_title_1('上海交通大学第四人民医院'))
        content.append(Graphics.draw_title_2('超声检查报告'))

        content.append(Graphics.draw_line())
        content.append(Spacer(0, 0.2 * cm))
        content.append(Graphics.draw_table([["姓名：锟铐烫", "性别：男", "年龄：22岁"],
                                            ["开单科室：CV", "床号：/", "卡号：/"]], page_width))

        content.append(Graphics.draw_line())
        content.append(Spacer(0, 0.5 * cm))
        content.append(Graphics.draw_title_3('超声图像：'))

        content.append(Graphics.draw_img_row(self.image, self.calcification, page_width))
        content.append(Spacer(0, 0.5 * cm))

        content.append(Graphics.draw_title_3('超声所见：'))
        content.append(Graphics.draw_text(self.desc, 20))
        content.append(Spacer(0, 0.1 * cm))
        content.append(Graphics.draw_text(self.generate_content(), 20))
        content.append(Spacer(0, 0.5 * cm))

        content.append(Graphics.draw_title_3('超声提示：'))
        content.append(Graphics.draw_text(f"C-TIRADS等级：{self.cal_CTIRAD()}", 20))
        content.append(Graphics.draw_text(self.ctirad, 20))

        signature = cv2.imread(random.choice(signatures))
        content.append(Graphics.draw_signature([["", "", "", "", "书写医师：",
                                                 convert_to_Image(signature, 30)]], page_width))

        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
        doc = SimpleDocTemplate(pdf_name, pagesize=A4, leftMargin=self.margin, rightMargin=self.margin,
                                topMargin=55)
        doc.build(content, onFirstPage=self.my_first_page, onLaterPages=self.my_later_pages)
