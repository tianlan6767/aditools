import os
import time
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from tqdm import tqdm
from typing import List
from evalmodel.data_info import ModelInfo
from evalmodel.data_analyzer import ModelAnalyzer
from evalmodel.utils.comm import sort_dict


__all__ = ['gen_model_report']


def _ppt_to_pdf(in_path, out_path='') -> str:
    """
        将PPT转换为PDF

        对Ubuntu系统不友好，需要一定的安装环境
    :param in_path: 输入文件
    :param out_path: 输出文件
    :return: 输出文件路径
    """

    from win32com.client import Dispatch
    powerpoint = Dispatch("Powerpoint.Application")
    powerpoint.Visible = 1
    slides = powerpoint.Presentations.Open(in_path)
    save_file = out_path if out_path else in_path.replace('.pptx', '.pdf')
    slides.ExportAsFixedFormat(save_file, 2, PrintRange=None)
    slides.Close()
    powerpoint.Quit()
    return save_file


def gen_model_report(analyzer: ModelAnalyzer, model_infos: List[ModelInfo], reporter: str = '',
                     reporting_format='pptx'):
    """
        生成模型评价报告

    :param analyzer: 模型分析器
    :param model_infos: 模型
    :param reporter: 创建人
    :param reporting_format: 报告格式 ['pptx', 'pdf']
    :return: None
    """

    if len(model_infos) == 0:
        return
    assert reporting_format in ['pptx', 'pdf'], reporting_format

    def bold_text(paragraph, text, size=Pt(26), rgb=RGBColor(0, 88, 159)) -> None:
        """
            凸出显示文本

        :param paragraph: PPT段落对象
        :param text: 显示文本
        :param size: 尺寸
        :param rgb: 颜色
        :return: None
        """

        _run = paragraph.add_run()
        _run.text = text
        _font = _run.font
        _font.name = 'Arial'
        _font.size = size
        _font.color.rgb = rgb

    def view_placeholder(slide):
        """查看占位符的类型和序列号"""
        for s in slide.placeholders:
            print(s, s.element.ph_idx)

    is_ok_test = model_infos[0].dft_num == 0
    ppt_num = 3 + len(model_infos) * (1 if is_ok_test else 2) + (0 if is_ok_test else 1)
    pro_bar = tqdm(range(ppt_num), desc='生成模型评估报告', ncols=90)

    # 获取模板PPT
    prs = Presentation(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'template/eva-template.pptx'))
    # 添加幻灯片首页
    slide_layout0 = prs.slide_layouts[0]
    slide0 = prs.slides.add_slide(slide_layout0)
    # 设置标题和副标题文本
    title = slide0.shapes.title
    subtitle = slide0.placeholders[1]
    title.text = '模型分析报告'
    subtitle.text = f'生成时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n' \
                    f'创建人：{reporter}'

    # 添加幻灯片，正文模块，根据实际需求选择布局版式

    # *************1. 测试数据*****************
    pro_bar.set_postfix(process='测试数据')

    # 添加幻灯片
    slide_layout1 = prs.slide_layouts[1]
    slide1 = prs.slides.add_slide(slide_layout1)

    # 添加标题
    title = slide1.placeholders[0]
    title.text = "1 测试数据"

    # 添加正文内容
    content = slide1.placeholders[1]
    ft = content.text_frame
    # ft.clear()
    p = ft.paragraphs[0]
    run = p.add_run()
    run.text = '（1) 缺陷实例'

    # 重点强调的内容
    bold_text(p, f'{model_infos[0].dft_num}个')

    # 继续添加其他内容
    run = p.add_run()
    run.text = f"\n（2) OK图像"
    bold_text(p, f'{model_infos[0].get_img_num("ok")}张')
    run = p.add_run()
    run.text = f"，NG图像"
    bold_text(p, f'{model_infos[0].get_img_num("ng")}张')
    run = p.add_run()
    run.text = f"，共"
    bold_text(p, f'{model_infos[0].get_img_num("both")}张')
    pro_bar.update()

    # *************2. 分析结论*****************
    pro_bar.set_postfix(process='分析结论')

    slide_layout2 = prs.slide_layouts[1]
    slide2 = prs.slides.add_slide(slide_layout2)
    title = slide2.placeholders[0]
    title.text = "2 分析结论"

    content = slide2.placeholders[1]
    ft = content.text_frame
    p = ft.paragraphs[0]
    run = p.add_run()
    run.text = '（1) 最优检出模型为'
    bold_text(p, f'{model_infos[0].model_key}.pth')
    run = p.add_run()
    run.text = f"，缺陷检出率为"
    bold_text(p, f'{model_infos[0].dft_chk_rate:.2%}')
    run = p.add_run()
    run.text = f"，图像检出率为"
    bold_text(p, f'{model_infos[0].img_chk_rate:.2%}')
    if len(model_infos) > 1:
        run = p.add_run()
        run.text = '\n（2) 次优检出模型为'
        bold_text(p, f'{model_infos[1].model_key}.pth')
        run = p.add_run()
        run.text = f"，缺陷检出率为"
        bold_text(p, f'{model_infos[1].dft_chk_rate:.2%}')
        run = p.add_run()
        run.text = f"，图像检出率为"
        bold_text(p, f'{model_infos[1].img_chk_rate:.2%}')
    pro_bar.update()

    # ***************3. 图像详情******************
    pro_bar.set_postfix(process='图像详情')

    slide_layout3 = prs.slide_layouts[9]
    slide3 = prs.slides.add_slide(slide_layout3)
    title = slide3.placeholders[0]
    title.text = "3 图像检出"

    img_data = analyzer.get_image_hist([model_info.model_key for model_info in model_infos])
    img_path = os.path.join(analyzer.work_space, f'hist_images/{len(img_data)}_images.jpg')
    picture_placeholder = slide3.placeholders[1]
    picture_placeholder.insert_picture(img_path)

    num = len(model_infos)
    num = num if num < 4 else 3
    message = f'图像检出最高前{num}的模型：\n'
    for m in sorted(model_infos, key=lambda info: info.img_chk_rate, reverse=True)[:num]:
        message += f'{m.model_key}: {m.img_chk_num}/{len(m.img_infos)} = {m.img_chk_rate:.2%}\n'
    content = slide3.placeholders[2]
    content.text = message
    pro_bar.update()

    # ***************4. 缺陷提升详情 (当纯OK数据评估时，不用输出这些信息) ******************
    if not is_ok_test:
        pro_bar.set_postfix(process='缺陷检出')
        pass

        slide_layout31 = prs.slide_layouts[9]
        slide31 = prs.slides.add_slide(slide_layout31)
        title = slide31.placeholders[0]
        title.text = "4 缺陷检出"

        dft_data = analyzer.get_defect_line([model_info.model_key for model_info in model_infos])
        img_path = os.path.join(analyzer.work_space, f'defect_line/{len(dft_data)}_defect.jpg')
        picture_placeholder = slide31.placeholders[1]
        picture_placeholder.insert_picture(img_path)

        # num = len(model_infos)
        # num = num if num < 4 else 3
        # message = f'图像检出最高前{num}的模型：\n'
        # for m in sorted(model_infos, key=lambda info: info.img_chk_rate, reverse=True)[:num]:
        #     message += f'{m.model_key}: {m.img_chk_num}/{len(m.img_infos)} = {m.img_chk_rate:.2%}\n'
        content = slide31.placeholders[2]
        content.text = '缺陷检出提升详情'
        pro_bar.update()

    # ***************4. 模型详情 ******************
    pro_bar.set_postfix(process='模型详情')

    for idx, model_info in enumerate(model_infos):
        # ***************4. 缺陷检出 (当纯OK数据评估时，不用输出这些信息)******************
        if not is_ok_test:
            slide_layout4 = prs.slide_layouts[8]
            slide4 = prs.slides.add_slide(slide_layout4)
            title = slide4.placeholders[0]
            title.text = f"{5 + idx}.1 模型检出"

            defect_infos = analyzer.get_defect_detail(model_info.model_key)[model_info.model_key]
            img_path = os.path.join(analyzer.work_space, f'defect_detail/{model_info.model_key}_MarkDetail.jpg')
            picture_placeholder = slide4.placeholders[1]
            picture_placeholder.insert_picture(img_path)

            chk_dft_infos = {}
            total_dft_infos = defect_infos['total']
            for label, num in defect_infos['check_mrk'].items():
                chk_dft_infos[label] = num / total_dft_infos[label]
            chk_dft_infos = sort_dict(chk_dft_infos, sort_by_key=False)
            num = len(chk_dft_infos)
            num = num if num < 5 else 4
            message = f'模型：{model_info.model_key}\n缺陷检出最低前{num}的类别：\n'
            for i, (label, chk_rate) in enumerate(chk_dft_infos.items()):
                if i >= num:
                    break
                chk_num = defect_infos['check_mrk'][label]
                total_num = total_dft_infos[label]
                message += f'{label}: {chk_num}/{total_num} = {chk_rate:.2%}\n'
            content = slide4.placeholders[2]
            content.text = message

            scores = analyzer.get_score_hist(model_info.model_key, 'check')[model_info.model_key]
            chk_mean_score = sum(scores) / len(scores) if len(scores) else 0

            ious = analyzer.get_iou_hist(model_info.model_key)
            mean_iou = sum(ious[model_info.model_key])
            iou_len = len(ious[model_info.model_key])
            mean_iou = mean_iou / iou_len if iou_len else 0
            content = slide4.placeholders[13]
            dft_num = model_info.dft_num
            content.text = f'\n模型推理结果{model_info.inf_num}个，详细分布如上图所示。其中，\n' \
                           f'真实缺陷检出{model_info.dft_chk_num}个，' \
                           f'缺陷检出率：{(model_info.dft_chk_num / dft_num) if dft_num else 0:.2%}，' \
                           f'平均得分：{chk_mean_score:.2f}，平均IOU：{mean_iou:.2f}\n' \
                           f'漏检{dft_num - model_info.dft_chk_num}个，' \
                           f'漏检率{((dft_num - model_info.dft_chk_num) / dft_num)  if dft_num else 0:.2%}\n'
            pro_bar.update()

        # ***************5. 缺陷过检******************
        slide_layout5 = prs.slide_layouts[8]
        slide5 = prs.slides.add_slide(slide_layout5)
        title = slide5.placeholders[0]
        title.text = f"{5 + idx}.{'1' if is_ok_test else '2'} 模型过检"

        over_infos = analyzer.get_over_pie(model_info.model_key)[model_info.model_key]
        img_path = os.path.join(analyzer.work_space, f'hist_overs/{model_info.model_key}_OverDefect.jpg')
        picture_placeholder = slide5.placeholders[1]
        picture_placeholder.insert_picture(img_path)

        scores = analyzer.get_score_hist(model_info.model_key, 'over')[model_info.model_key]
        over_mean_score = sum(scores) / len(scores) if len(scores) else 0
        over_max_class, over_max_num = [], 0
        for cls, num in over_infos.items():
            if over_max_num == 0:
                over_max_class.append(str(cls))
                over_max_num = num
            elif num == over_max_num:
                over_max_class.append(str(cls))
            else:
                break
        content = slide5.placeholders[2]
        content.text = f'模型：{model_info.model_key}\n' \
                       f'缺陷过检总数：{model_info.dft_over_num}\n' \
                       f'缺陷过检最高类别：{", ".join(over_max_class) if len(over_max_class) else "None"} \n' \
                       f'缺陷过检平均得分：{over_mean_score:.2f}\n'

        defect_infos = analyzer.get_defect_detail(model_info.model_key)[model_info.model_key]
        content = slide5.placeholders[13]
        content.text = f'缺陷过检分布： defectNum/imageNum \n' \
                       f'OK: {defect_infos["dft_over_ok_num"]}/{defect_infos["img_over_ok_num"]} \n' \
                       f'NG: {defect_infos["dft_over_ng_num"]}/{defect_infos["img_over_ng_num"]}'
        pro_bar.update()

    # ***************生成评估报告******************
    pro_bar.close()
    save_file = os.path.join(analyzer.work_space, '模型分析报告.pptx')
    try:
        prs.save(save_file)
    except PermissionError:
        print(f'报告已打开，将重名命名生成的报告')
        cur_time = time.strftime('%H%M%S', time.localtime(time.time()))
        save_file = os.path.join(analyzer.work_space, f'模型分析报告_{cur_time}.pptx')
        prs.save(save_file)
    finally:
        time.sleep(0.05)
        if reporting_format == 'pdf':
            temp = save_file
            save_file = _ppt_to_pdf(save_file)
            os.remove(temp)
        print(f'报告已生成，请打开PPT文件查看。"{save_file}:0"')
