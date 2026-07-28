"""
规范文本爬取可视化工具
基于 Gradio 构建，输入网址即可爬取规范文本并下载 txt 文件
"""

import os
import time
import random

import gradio as gr
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup


# ============================================================
# 爬虫核心逻辑（改编自 spider/requeset.py）
# ============================================================

def scrape_specification(url: str, title: str):
    """
    爬取规范文本，返回 (状态信息, 文本预览, 文件路径)

    Args:
        url:   规范页面网址
        title: 规范名称（用作文件名）

    Yields:
        (status_msg, text_preview, file_path)
    """
    # ----- 输入校验 -----
    if not url or not url.strip():
        yield "❌ 请输入有效的网址", "", None
        return

    if not title or not title.strip():
        yield "❌ 请输入规范名称", "", None
        return

    url = url.strip()
    title = title.strip()

    # ----- 准备保存路径 -----
    base_path = os.path.dirname(__file__)
    file_dir = os.path.join(base_path, "规范文件")
    os.makedirs(file_dir, exist_ok=True)
    txt_path = os.path.join(file_dir, f"{title}.txt")

    yield f"⏳ 正在初始化浏览器...", "", None

    # ----- 配置 Chrome -----
    option = ChromeOptions()
    option.add_experimental_option("excludeSwitches", ["enable-automation"])
    option.add_argument("--ignore-certificate-errors")
    option.add_argument("--ignore-ssl-errors")
    option.add_argument("--allow-running-insecure-content")
    option.add_argument("--disable-web-security")
    # 无头模式（后台运行，不弹窗）
    option.add_argument("--headless=new")
    option.add_argument("--no-sandbox")
    option.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=option)
        driver.maximize_window()

        yield f"🌐 正在访问页面: {url}", "", None
        driver.get(url)
        time.sleep(random.uniform(2, 4))

        yield "📄 正在解析页面内容...", "", None

        # ----- 解析页面 -----
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        lines = []
        lines.append(f"{title}\n")

        content_divs = soup.find_all("div", id=True)

        for div in content_divs:
            # 章节标题
            title_element = div.find("div")
            if title_element:
                chapter_title = title_element.get_text().strip()
                if chapter_title:
                    lines.append(f"\n{chapter_title}")
                    lines.append("=" * 50)

            # 段落内容
            paragraphs = div.find_all("p")
            for p in paragraphs:
                text = p.get_text().strip()
                if text:
                    if "条文说明" in text:
                        lines.append("\n【条文说明】")
                        text = text.replace("条文说明", "").strip()
                    lines.append(text)

            lines.append("")  # div 间空行

        full_text = "\n".join(lines)

        # ----- 写入文件 -----
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # ----- 生成预览（前 3000 字）-----
        preview = full_text[:3000]
        if len(full_text) > 3000:
            preview += f"\n\n... （共 {len(full_text)} 字符，此处仅显示前 3000 字）"

        yield f"✅ 爬取完成！共 {len(full_text)} 字符，已保存至: {txt_path}", preview, txt_path

    except Exception as e:
        yield f"❌ 爬取出错: {str(e)}", "", None

    finally:
        if driver:
            driver.quit()


# ============================================================
# Gradio 可视化界面
# ============================================================

def build_ui():
    with gr.Blocks(title="规范文本爬取工具", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 📘 规范文本爬取工具

            输入规范页面的 **网址** 和 **规范名称**，一键爬取并下载规范全文。
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                url_input = gr.Textbox(
                    label="🔗 规范页面网址",
                    placeholder="例如: http://zlglpt.com/book/book_view.aspx?id=3218",
                    lines=1,
                )
            with gr.Column(scale=1):
                title_input = gr.Textbox(
                    label="📛 规范名称",
                    placeholder="例如: 公路钢混组合桥梁设计与施工规范",
                    lines=1,
                )

        with gr.Row():
            scrape_btn = gr.Button("🕷️ 开始爬取", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg")

        status_output = gr.Textbox(
            label="📌 状态",
            placeholder="等待输入...",
            lines=1,
            interactive=False,
        )

        preview_output = gr.Textbox(
            label="📖 文本预览",
            placeholder="爬取结果将在此处显示...",
            lines=20,
            max_lines=30,
            interactive=False,
        )

        download_output = gr.File(label="📥 下载 txt 文件", visible=True)

        # ----- 按钮事件 -----
        scrape_btn.click(
            fn=scrape_specification,
            inputs=[url_input, title_input],
            outputs=[status_output, preview_output, download_output],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", None),
            inputs=[],
            outputs=[url_input, title_input, status_output, preview_output, download_output],
        )

        gr.Markdown("---\n💡 提示：爬取需要 Chrome 浏览器和 ChromeDriver，请确保已安装。")

    return app


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,  # 自动打开浏览器
    )
