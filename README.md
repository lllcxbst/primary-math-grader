# 小学数学智能批改 Demo

完整说明请参阅 [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)。

一个可直接上传小学数学题目照片的网页 Demo。它通过硅基流动的
`Qwen/Qwen3-VL-32B-Instruct` 多模态模型：

- 识别题目和学生作答
- 逐题判断“正确 / 错误 / 未作答 / 无法辨认”
- 对错题给出具体原因、正确答案和订正步骤
- 支持一次上传最多 8 张图片
- 显示分阶段进度、等待秒数和最终总耗时
- 流式取得服务端 Trace ID，并区分连接超时与模型读取超时
- 生成不含密钥和图片内容的本地诊断日志 `api_requests.log`
- 支持快速/详细两种批改模式
- 可选择主流多模态模型或填写自定义模型 ID
- 支持补充参考答案和教师说明
- 支持下载 Markdown 格式的批改结果

## 1. 配置 API 密钥

在项目根目录复制配置模板：

```powershell
Copy-Item .env.example .env
```

打开 `.env`，将下面一行改成你自己的密钥：

```dotenv
SILICONFLOW_API_KEY=sk-你的密钥
```

`.env` 已被 `.gitignore` 忽略，不会被 Git 提交。不要把真实密钥写入
`app.py`、截图或公开仓库。

## 2. Windows 一键准备环境

```powershell
.\setup.ps1
```

如果系统阻止脚本执行，可使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## 3. 启动

```powershell
.\run.ps1
```

浏览器会打开 `http://localhost:8501`。上传清晰的题目照片，选择年级，
点击“开始批改”即可。

也可以手动启动：

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## 4. 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试会模拟 API 响应，不会消耗硅基流动额度。

## 拍照建议

- 题目和学生答案必须完整入镜。
- 手机尽量正对纸面，避免透视变形、反光和阴影。
- 字迹较小时，一张照片少拍几道题。
- 应用题要拍全题干、单位和学生答句。
- AI 批改仍可能出错，重要作业请由老师或家长复核。

## 速度建议

- 日常单题优先选择 `Qwen/Qwen3-VL-8B-Instruct` 和“快速批改”。
- 复杂几何、图表或多步应用题可使用 32B/Thinking 模型，但耗时和费用更高。
- 普通单题最长边设为 1600～2000 即可；整页小字试卷再使用 3000～3584。
- 中间 20%～92% 是基于等待时间的估算，因为服务端接口不返回真实推理百分比。

## 项目结构

```text
.
├── app.py                 # Streamlit 网页
├── exam_analyzer.py       # 图片预处理、提示词和 API 调用
├── tests/
│   └── test_exam_analyzer.py
├── requirements.txt
├── setup.ps1              # 创建虚拟环境并安装依赖
├── run.ps1                # 启动网页
├── .env.example
└── .gitignore
```

## API 说明

程序调用：

```text
POST https://api.siliconflow.cn/v1/chat/completions
```

图片会在本地进行方向校正和尺寸压缩，再以 Base64 Data URL 发送。
密钥仅通过请求头中的 Bearer Token 传递，不会显示在页面或日志中。
