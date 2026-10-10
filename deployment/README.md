# 阿里云 ECS 网站发布包

这个目录对应“网站发布”，与 `competition/*cloud*.zip` 的 RCWA 批量计算包分开。

当前竞赛发布入口：展示页 `http://47.111.14.70/`，交互页 `http://47.111.14.70/app/`，代码和文档见 `https://github.com/qiaoanqi/metasurface-web`，离线包见 GitHub Release `competition-web-v5.0.0`。当前 ECS 对齐包为 `ai_metasurface_web_bundle_20261011_v5.zip`，包的 SHA-256 记录在包外的部署说明和部署回执中。二维码绑定展示页；展示页内可进入交互页；公网 IP 变更后需要重新生成二维码并复测。

部署后：

- `http://服务器公网IP/`：网站展示页；
- `http://服务器公网IP/app/`：Streamlit 完整交互台；
- `http://服务器公网IP/health`：Nginx 存活检查。

## ECS 安装

建议在 Ubuntu 22.04/24.04、4 vCPU、16 GB RAM、100 GB SSD 上运行；这套配置用于校赛展示更稳。少量单人演示可降到 2 vCPU/8 GB/80 GB，但不建议作为答辩当天唯一实例。安全组至少放行 TCP `80`，申请 HTTPS 后再放行 `443`。`8501` 只绑定回环地址，不需要对公网开放。

把本 ZIP 解压到 `/opt/ai-metasurface-web`，然后执行：

```bash
cd /opt/ai-metasurface-web
sudo bash deployment/install.sh
```

安装脚本会创建独立系统用户、Python 虚拟环境、安装 `requirements-web.txt`，并启用。依赖文件会按 Python 版本选择 ONNX Runtime：Ubuntu 22.04/Python 3.10 使用 `onnxruntime==1.23.2`，Python 3.11 及以上使用 `1.26.0`。

- `metasurface-streamlit.service`
- `nginx` 静态页和 `/app/` 反向代理

检查：

```bash
sudo systemctl status metasurface-streamlit --no-pager
bash deployment/scripts/healthcheck.sh http://127.0.0.1
```

首次启动会按需加载 ONNX 模型和 21,088 条已审核 TiO2/SiO2/air 参考记录。参考库只做精确命中，不做最近邻、插值或训练。发布包同时包含注册模型的梯度搜索权重、RL 表、分析证据及图像资源；ECS 使用 CPU PyTorch 支持相应功能。

## 运行边界

这个包是竞赛展示/交互运行时，不包含 Paper1/Paper2 控制面、训练集、holdout、active pool 或长时间 RCWA 任务。部署成功只代表网页和交互服务可访问，不代表代理模型全域精度、RCWA 收敛或实验验证。
