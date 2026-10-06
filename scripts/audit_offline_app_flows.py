"""Run real Streamlit app actions in a shipped runtime without model stubs."""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    os.chdir(runtime)
    sys.path.insert(0, str(runtime))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    import numpy as np
    import torch
    from streamlit.testing.v1 import AppTest
    from ui_inverse_contracts import serialize_inverse_run
    from ui_analysis_snapshots import ANALYSIS_SESSION_KEYS, validate_snapshot_record
    torch.set_num_threads(1)
    software = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(runtime.glob('*.py'))}
    at = AppTest.from_file(str(runtime / 'app.py'), default_timeout=180)
    checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run():
        at.run(timeout=180)
        assert not at.exception, [item.message for item in at.exception]
        assert not at.error, [item.value for item in at.error]

    def button(label=None, key=None):
        item = at.button(key=key) if key else next(b for b in at.button if b.label == label)
        assert not item.disabled, f'Action disabled: {label or key}'
        item.click()
        run()

    def control(collection, label, value):
        next(item for item in getattr(at, collection) if item.label == label).set_value(value)
        run()

    def record(name, action):
        begin = time.perf_counter()
        try:
            detail = action()
            row = dict(name=name, status='pass', detail=detail)
        except Exception as exc:
            row = dict(name=name, status='fail', error=f'{type(exc).__name__}: {exc}')
        row['elapsed_s'] = round(time.perf_counter()-begin, 3)
        checks.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def inverse(label, method, apply_key=None):
        button(label)
        current = at.session_state['_inverse_run']
        assert current.method_id == method
        exports = serialize_inverse_run(current, current.context)
        candidates = exports.payload['candidates']
        assert candidates
        assert all(np.isfinite(c['delta_e2000']) for c in candidates)
        for suffix, text in (('json', exports.json_text), ('csv', exports.csv_text)):
            (args.output.parent / f'offline_flow_{method}_20261006.{suffix}').write_text(text, encoding='utf-8')
        if apply_key:
            button(key=apply_key)
        return dict(candidates=len(candidates), applied=bool(apply_key),
                    best_color=candidates[0]['predicted_hex'])

    def analysis(kind, key):
        button(key=key)
        snapshot = validate_snapshot_record(at.session_state[ANALYSIS_SESSION_KEYS[kind]])
        assert snapshot.payload['status'] == 'available', snapshot.payload.get('reason')
        payload = snapshot.to_record()
        (args.output.parent / f'offline_flow_{kind}_{snapshot.context.structure_type}_20261006.json').write_text(
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
        return dict(payload_sha256=snapshot.payload_sha256,
                    route=snapshot.context.route_id)

    record('startup/all_five_pages', lambda: (run() or dict(
        pages=[tab.label for tab in at.tabs], buttons=len(at.button))))
    record('single/smart_search_apply_and_exports', lambda: inverse(
        '开始搜索 · 智能网格', 'smart', 'apply_sg_result'))
    record('single/gradient_search_apply_and_exports', lambda: inverse('单柱梯度','single','apply_gd_result'))
    record('single/rl_search_apply_and_exports', lambda: inverse('RL Q-learning','rl','apply_rl_result'))
    record('single/sensitivity', lambda: analysis('sensitivity','run_sensitivity_analysis'))
    record('single/dh_mapping', lambda: analysis('mapping','run_mapping_analysis'))
    record('single/angle_scan', lambda: analysis('angle','run_angle_analysis'))

    def preview_routes():
        control('checkbox','启用 ML 代理模型（快速候选预测）',False)
        control('selectbox','偏振','TM (p-pol)')
        control('number_input','精确输入 角度',30.)
        control('checkbox','启用角谱远场传播 (Angular Spectrum)',True)
        labels = [item.label for item in at.number_input]
        assert '精确 NA' in labels
        control('checkbox','启用角谱远场传播 (Angular Spectrum)',False)
        control('number_input','精确输入 角度',0.)
        control('selectbox','偏振','TE (s-pol)')
        return dict(tm_and_oblique_and_far_field=True)
    record('single/physical_tm_angle_and_far_field',preview_routes)

    def reference():
        button('加载可复核示例')
        assert any('140/281/407' in item.value for item in at.success)
        inputs = {item.label:item.value for item in at.number_input}
        assert [inputs[label] for label in ('精确输入 D','精确输入 H','精确输入 P')] == [140,281,407]
        label = next(b.label for b in at.button if b.label.startswith('开始搜索 ·'))
        result = inverse(label,'reference','apply_reference_result_1')
        return dict(**result, exact_geometry=[140,281,407], conditions='TM/0deg')
    record('reference/load_verified_sample', reference)

    def dual():
        control('radio','📏 结构类型','双柱')
        return inverse('开始搜索 · 双柱解析', 'dual_physical', 'apply_dual_physical_result')
    record('dual/analytical_search_apply_and_exports', dual)
    record('dual/sensitivity', lambda: analysis('sensitivity','run_sensitivity_analysis'))
    record('dual/angle_scan', lambda: analysis('angle','run_angle_analysis'))

    def fp():
        control('radio','📏 结构类型','FP 腔（Fabry-Pérot）')
        control('selectbox','反射镜类型','介质 DBR (TiO2/SiO2)')
        first = inverse('开始搜索 · FP 腔搜索', 'fp', 'fp_apply_0')
        button('开始搜索 · FP 腔搜索')
        assert any('命中缓存' in item.value for item in at.success)
        return dict(**first, second_search_cache_hit=True)
    record('fp/dbr_search_apply_exports_and_cache', fp)
    record('fp/angle_scan', lambda: analysis('angle','run_angle_analysis'))

    def ag():
        control('selectbox','反射镜类型','金属 Ag (减色)')
        assert any('FP cavity TMM' in item.value for item in at.markdown)
        return dict(preview=True, inverse_policy='dbr_only')
    record('fp/ag_preview', ag)

    def themes():
        control('radio','📏 结构类型','单柱')
        for value in ('浅色','高对比','深色'):
            at.session_state['ui_theme_control'] = value
            run()
        for value in ('靛蓝','青绿色','紫罗兰','琥珀橙'):
            at.session_state['ui_accent_control'] = value
            run()
        return dict(themes=3, accents=4)
    record('themes/session_switches',themes)

    def model_difference():
        control('selectbox','偏振','TE (s-pol)')
        control('checkbox','启用 ML 代理模型（快速候选预测）',True)
        button('▶ 运行模型间差异分析')
        snapshot = validate_snapshot_record(at.session_state[ANALYSIS_SESSION_KEYS['difference']])
        assert snapshot.payload['status'] == 'available', snapshot.payload.get('reason')
        return dict(payload_sha256=snapshot.payload_sha256)
    record('analysis/model_difference',model_difference)

    def benchmark():
        button('▶ 运行当前可用方法基准')
        rows = at.session_state['_bench_cache']
        assert rows and not [row for row in rows if row.status == 'error']
        return {row.method_id:row.status for row in rows}
    record('analysis/method_benchmark',benchmark)

    def gamut():
        control('checkbox','显示模型路线色域对比图',True)
        assert not [w.value for w in at.warning if '色域凸包不可用' in w.value]
        control('checkbox','显示模型路线色域对比图',False)
        return dict(routes=4, plotting_completed=True)
    record('analysis/gamut_comparison',gamut)
    result = dict(schema='offline-real-app-flows-v1',runtime=str(runtime),
                  status='pass' if all(c['status']=='pass' for c in checks) else 'fail',
                  mocked_models=False, checks=checks, software_sha256=software)
    assert all(hashlib.sha256((runtime/name).read_bytes()).hexdigest() == digest
               for name,digest in software.items()), 'Runtime changed during audit'
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    temporary.replace(args.output)
    return 0 if result['status']=='pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
