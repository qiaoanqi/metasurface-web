# 部署到阿里云 ECS「科创包」

## 1. 本地发布包

文件：`dist/ai_metasurface_web_bundle_v1.zip`

本版 SHA-256（精简模型差异分析默认展示）：

```text
E05312F3364036C4542C86B5B008D4168C21D4FBD68BE0254D530DDC766E4044
```

包内 TiO2/SiO2/air 参考记录 SHA-256：

```text
01BD65E80DF2A3C55BD92316CCD7882CD050969057278DB885BD0EAE83BB8403
```

## 2. 上传位置

在 ECS 实例详情页进入“上传/下载文件”，选择本地 ZIP，目标路径填写：

```text
/root/ai_metasurface_web_bundle_v1_release_20261001.zip
```

注意：目标是一个以 `.zip` 结尾的普通文件，不要先创建同名目录，也不要把 ZIP 上传到 `/root/ai_metasurface_web_bundle_v1_release_20261001.zip/`。Cloud Assistant 的“发送文件”入口只适合几十 KB 的脚本；这个 168 MB 发布包应使用实例详情页的文件上传入口。

## 3. Cloud Assistant 执行部署命令

上传完成后，在同一台 ECS、同一地域执行下面的 Shell 命令。先校验包，再解压到临时目录；如果已有旧版，会自动改名保留，不覆盖删除。

```bash
set -eu

ZIP=/root/ai_metasurface_web_bundle_v1_release_20261001.zip
EXPECTED=E05312F3364036C4542C86B5B008D4168C21D4FBD68BE0254D530DDC766E4044
test -f "$ZIP"
printf '%s  %s\n' "$EXPECTED" "$ZIP" | sha256sum -c -

if ! command -v unzip >/dev/null 2>&1; then
    apt-get update
    apt-get install -y unzip
fi
STAGE=$(mktemp -d /opt/.ai-metasurface-web.stage.XXXXXX)
unzip -q "$ZIP" -d "$STAGE"
test -f "$STAGE/app.py"
test -f "$STAGE/static/index.html"
test -f "$STAGE/competition/tio2_air_reference_records_v1.jsonl"

if test -e /opt/ai-metasurface-web; then
    mv /opt/ai-metasurface-web "/opt/ai-metasurface-web.backup.$(date +%Y%m%d%H%M%S)"
fi
mv "$STAGE" /opt/ai-metasurface-web

cd /opt/ai-metasurface-web
bash deployment/install.sh
```

## 4. 验证

```bash
cd /opt/ai-metasurface-web
bash deployment/scripts/healthcheck.sh http://127.0.0.1
systemctl is-active --quiet metasurface-streamlit
systemctl is-active --quiet nginx
sha256sum competition/tio2_air_reference_records_v1.jsonl
```

浏览器访问：

```text
http://ECS公网IP/
http://ECS公网IP/app/
```

## 4.1 已上传旧包时的兼容修复

如果已经上传并解压了旧版发布包，且安装日志显示 `No matching distribution found for onnxruntime==1.26.0`，不要重新上传。Ubuntu 22.04 默认 Python 3.10，应在服务器上执行：

```bash
cd /opt/ai-metasurface-web
sed -i '/^onnxruntime==1.26.0$/c\onnxruntime==1.23.2; python_version < "3.11"\nonnxruntime==1.26.0; python_version >= "3.11"' requirements-web.txt
bash deployment/install.sh
/opt/ai-metasurface-web/.venv/bin/python -c 'import onnxruntime as ort; print("onnxruntime=" + ort.__version__)'
bash deployment/scripts/healthcheck.sh http://127.0.0.1
```

预期输出中的 ONNX Runtime 版本为 `1.23.2`，两个 systemd 服务应为 active。后续重新上传时使用本页顶部记录的新包 SHA256。

网站首页在线部署后，“打开交互台”按钮会自动指向同域 `/app/`；本地 `file://` 预览仍指向 `127.0.0.1:8512`。

## 5. 安全组

- 放行 TCP `80`；
- 若配置 HTTPS，再放行 TCP `443`；
- 不需要对公网放行 `8501`，Streamlit 只监听 `127.0.0.1:8501`。

若需要更新版本，重复上传新 ZIP，先换算新的 `EXPECTED`，再执行同一套临时目录流程；旧目录会以 `.backup.YYYYMMDDHHMMSS` 保留。
