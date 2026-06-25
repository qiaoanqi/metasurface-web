---
title: AI超表面结构色智能设计系统
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# AI超表面结构色智能设计系统

基于物理模型（Fano共振 + 米氏散射）与深度学习（ResMLP）的超表面纳米柱结构色正向预测与逆设计平台。

## 功能
- 🔬 实时颜色预览：调节直径D、高度H、周期P，实时观察颜色变化
- 🎯 逆设计：5种方法（网格搜索、RL、单/双柱梯度优化、AI智能寻色）
- 🖼️ 图案生成：上传图片生成结构色马赛克
- 📊 颜色映射：D-H参数空间颜色图谱
- 🌈 光谱分析：反射光谱与CIE色度图
- 🤖 AI分析：LLM智能解读颜色物理机理

## 云端地址
https://huggingface.co/spaces/qiaoanqi/metasurface-color-designer

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```
