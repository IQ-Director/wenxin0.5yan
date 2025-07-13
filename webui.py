#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的代码
import erniebot
import gradio as gr
import os
import shutil
from utils import ErniebotEmbeddings
from langchain_community.vectorstores import FAISS

token = ''
index_path='index'
def question_faiss(input_text):
    try:
        # 初始化 ErnieBot
        erniebot.api_type = "aistudio"
        erniebot.access_token = token
        
        # 加载嵌入和索引
        embedding = ErniebotEmbeddings(erniebot.access_token)
        index = FAISS.load_local(index_path, embedding, allow_dangerous_deserialization=True)
        
        # 准备查询和提示
        query = input_text  # 使用用户输入的文本作为查询
        k = 10
        search_type = 'similarity'  # mmr
        prompt = '请根据如下问题和相关知识文档给出回答。\n问题：%s\n参考：\n' % query
        references = []
        
        # 执行搜索并构建提示
        search_results = index.search(query, k=k, search_type=search_type)
        for i, doc in enumerate(search_results):
            prompt += '[%d] %s\n%s\n\n' % (i + 1, doc.metadata['source'], doc.page_content)
            reference = doc.metadata['source'].replace('docs-develop', 'https://www.msdmanuals.cn/professional/neurologic-disorders/stroke/overview-of-stroke')
            if reference not in references:
                references.append(reference)
        prompt += '\n回答：\n'
        
        # 调试提示内容
        print("Prompt:", prompt)
        
        # 调用 ErnieBot API
        response = erniebot.ChatCompletion.create(
            model='ernie-3.5',
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        return response.result

    except erniebot.errors.InvalidTokenError as e:
        return "无效的 token，请检查 token 是否正确"
    except ValueError as e:
        # 如果解析 JSON 失败，打印响应内容
        return f"响应解析失败，请检查 API 响应格式。\nFailed to parse JSON:, {e}"
    except Exception as e:
        # 捕获其他异常并打印错误信息
        print("Error:", e)
        return "发生错误，请检查代码和 API 配置"

def question_without(input_text):
    erniebot.api_type = "aistudio"
    erniebot.access_token = token
    try:
        response = erniebot.ChatCompletion.create(
            model='ernie-3.5',
            messages=[{
                "role": "user",
                "content": f"{input_text}"
            }]
        )
    except erniebot.errors.InvalidTokenError as e:
        return "无效的token，请检查token是否正确"
    print(response.result)
    return response.result

question = question_without
examples = [
    ["今天的天气如何？"],
    ["今天建议出行吗？"],
    ['可以帮我解决千禧年难题吗，奖金分你一半。']
]
def load_token(text):
    # 在这里处理输入的文本，可以将其存储在变量中
    global token
    token = text
    return token

def clear_directory(directory):
    """清空目标目录"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
# 自定义 CSS 样式隐藏 .textbox-label 类的元素


def create_ui_and_launch():
    with gr.Blocks(title="文心0.5言", theme=gr.themes.Soft()) as block:
        with gr.Row():
            gr.HTML("""<h1 align="center">文心0.5言</h1>""")
        with gr.Accordion("【使用说明】（点击展开/折叠）", open=False):
            gr.Markdown(
                """
                ERNIE Bot SDK是文心&飞桨官方提供的Python软件开发工具包，其提供便捷易用的Python接口，可调用文心一言大模型能力，完成包含文本创作、通用对话、语义向量、AI作图在内的多项任务。ERNIE Bot SDK代码在GitHub上开源，欢迎大家进入[repo](https://github.com/PaddlePaddle/ERNIE-Bot-SDK)查看源码和使用文档，如果遇到问题也可以提出issue。
                
                在[访问令牌页面](https://aistudio.baidu.com/usercenter/token)复制access token，并粘贴到“Token”文本框中。
                """
            )
        token_input = gr.Textbox(label="token", placeholder="输入token...",type="password")
        button = gr.Button("确认")
        button.click(fn=load_token, inputs=token_input)
        switch = gr.Checkbox(label="是否使用文本向量库")
        def process_switch(is_checked):
            global question
            if is_checked:
                question = question_faiss
                return gr.update(visible=True), gr.update(visible=True)
            else:
                question = question_without
                return gr.update(visible=False), gr.update(visible=False)
        def process_file(files):
            for file in files:
                # 获取上传文件的路径
                file_path = file.name
                # 获取文件夹路径
                folder_path = os.path.dirname(file_path)
                # 清空目标目录
                # clear_directory(folder_path)
                # 确保目标目录存在
                os.makedirs(folder_path, exist_ok=True)
                # 目标路径
                target_path = os.path.join(r'/home/aistudio/index', os.path.basename(file_path))
                shutil.copy(file_path, target_path)
            return "上传完毕"
        # 文件选择框
        file_input = gr.File(label="选择文件", file_count="multiple",file_types=[".faiss",".pkl",".index"],visible=False)
        output = gr.Textbox(label="上传提醒", placeholder="请上传.fasis/.index文件与.对应的.pkl文件。",visible=False)
        switch.change(fn=process_switch, inputs=switch, outputs=[file_input, output])
        file_input.change(fn=process_file, inputs=file_input, outputs=output)
        
        create_chat_completion_tab()
    block.launch()

def create_chat_completion_tab():
    with gr.Tab('你的对话(Chat)') as chat_completion_tab:
        with gr.Row():
            with gr.Column(scale=4):
                input_text = gr.Textbox(label="Q", lines=4, placeholder="请输入...")
                output_text = gr.Textbox(label="A", lines=4, placeholder="")

                with gr.Row():
                    send_btn = gr.Button("提问")
                    clear_btn = gr.Button("清空")
                    send_btn.click(question, inputs=[input_text], outputs=[output_text])
                    clear_btn.click(lambda _: (None, None), inputs=clear_btn, outputs=[input_text, output_text])
                    gr_examples = gr.Examples(examples=examples, inputs=[input_text],
                                              label="输入示例 (点击选择例子)",
                                              examples_per_page=20)


if __name__ == '__main__':
    create_ui_and_launch()
    
