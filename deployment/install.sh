#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/ai-metasurface-web}"
SERVICE_NAME="metasurface-streamlit"

if [[ "${EUID}" -ne 0 ]]; then
    echo "请用 sudo 执行：sudo bash deployment/install.sh" >&2
    exit 2
fi

if [[ "${PACKAGE_ROOT}" != "${INSTALL_ROOT}" ]]; then
    echo "请先把发布包解压到 ${INSTALL_ROOT}，当前包路径为 ${PACKAGE_ROOT}" >&2
    exit 2
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3 python3-venv python3-pip nginx curl ca-certificates unzip

if ! id -u metasurface >/dev/null 2>&1; then
    useradd --system --home-dir "${INSTALL_ROOT}/.runtime-home" --create-home --shell /usr/sbin/nologin metasurface
fi

python3 -m venv "${INSTALL_ROOT}/.venv"
"${INSTALL_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${INSTALL_ROOT}/.venv/bin/pip" install --requirement "${INSTALL_ROOT}/requirements-web.txt"

install -d -m 0750 -o metasurface -g metasurface "${INSTALL_ROOT}/.runtime-home"
chown -R metasurface:metasurface "${INSTALL_ROOT}"
# The release stage is created by mktemp with mode 0700. Nginx must be able to
# traverse the application root to serve the public showcase files.
chmod 0755 "${INSTALL_ROOT}"

install -m 0644 "${INSTALL_ROOT}/deployment/systemd/metasurface-streamlit.service" \
    "/etc/systemd/system/${SERVICE_NAME}.service"
install -m 0644 "${INSTALL_ROOT}/deployment/nginx/metasurface.conf" \
    /etc/nginx/sites-available/metasurface.conf
ln -sfn /etc/nginx/sites-available/metasurface.conf /etc/nginx/sites-enabled/metasurface.conf
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}.service"
systemctl enable --now nginx
systemctl restart "${SERVICE_NAME}.service"
systemctl reload nginx

echo "部署完成："
echo "  展示页: http://$(hostname -I | awk '{print $1}')/"
echo "  交互台: http://$(hostname -I | awk '{print $1}')/app/"
echo "  服务状态: systemctl status ${SERVICE_NAME} --no-pager"
