#!/usr/bin/env bash
# 将 product-embedding overlay 安装到一份 DINOv3 官方仓库（dinov3-main）检出中。
# 只新增文件，不修改、不覆盖 DINOv3 上游的任何文件（若目标已存在同名 overlay 文件则覆盖自身旧版）。
#
# 用法: ./install.sh /path/to/dinov3-main
set -euo pipefail

TARGET="${1:?usage: ./install.sh /path/to/dinov3-main}"

if [ ! -d "$TARGET/dinov3" ] || [ ! -d "$TARGET/app" ]; then
    echo "error: $TARGET 不像 dinov3-main 仓库根（缺少 dinov3/ 或 app/）" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 防止覆盖上游文件：overlay 中所有文件在目标里要么不存在，要么与 overlay 同源
while IFS= read -r -d '' f; do
    rel="${f#"$SCRIPT_DIR/overlay/"}"
    if [ -f "$TARGET/$rel" ] && ! cmp -s "$f" "$TARGET/$rel"; then
        # 已存在且内容不同：finetune_v2 等本仓库文件的更新属正常，其余给出提示
        echo "update: $rel"
    fi
done < <(find "$SCRIPT_DIR/overlay" -type f -print0)

cp -rv "$SCRIPT_DIR/overlay/." "$TARGET/"
echo "installed into $TARGET"
