"""Exercise shipped inference and design features without training or RCWA runs.

Run against an extracted offline package, never implicitly against the checkout.
The numerical parity check is a software check, not a physical accuracy claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    import numpy as np
    import torch
    import ml_module as ml
    import torch_model as tm
    from color_utils import delta_e2000, rgb_to_lab

    assert Path(ml.__file__).resolve().parent == runtime
    assert Path(tm.__file__).resolve().parent == runtime
    torch.set_num_threads(1)
    np.random.seed(20261006)
    torch.manual_seed(20261006)
    checks = []
    started = time.time()

    def check(name, action):
        begin = time.perf_counter()
        try:
            detail = action()
            row = dict(name=name, status='pass', detail=detail)
        except Exception as exc:
            row = dict(name=name, status='fail', error=f'{type(exc).__name__}: {exc}')
        row['elapsed_s'] = round(time.perf_counter() - begin, 3)
        checks.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def finite_unit(values, shape):
        values = np.asarray(values)
        assert values.shape == shape, values.shape
        assert np.isfinite(values).all()
        assert values.min() >= -1e-6 and values.max() <= 1 + 1e-6
        return values

    weights = sorted((runtime / 'models').glob('*.pt'))
    tables = sorted((runtime / 'models').glob('rl*.npy'))
    frozen = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in weights + tables}
    software = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(runtime.glob('*.py'))}
    assert ml.init_rcwa_ml(), 'RCWA model registry did not load'
    target = np.array([0.3, 0.4, 0.8])

    for material, mc in ml.MATERIAL_CODES.items():
        for substrate, sc in ml.SUBSTRATE_CODES.items():
            def exercise_pair(material=material, substrate=substrate, mc=mc, sc=sc):
                paths = ml.rcwa_torch_artifact_paths(material, substrate)
                model = tm._load_rcwa_torch_model('cpu', material, substrate)
                d = torch.tensor([100., 180., 250., 320.])
                h = torch.tensor([150., 300., 450., 550.])
                p = torch.tensor([220., 400., 450., 500.])
                x = tm._rcwa_input_batch(d, h, p, mc, sc)
                with torch.no_grad():
                    pt = finite_unit(model(x).numpy(), (4, 81))
                sessions = ml._get_rcwa_sessions(material, substrate)
                assert len(sessions) == len(paths)
                onnx = finite_unit(np.mean([
                    ml._predict_onnx_batch(session, x.numpy(), 81)
                    for session in sessions], axis=0), (4, 81))
                error = float(np.max(np.abs(pt - onnx)))
                assert error <= 1e-5, error
                result = tm.inverse_design_rcwa(
                    torch.tensor(target, dtype=torch.float32), n_restarts=3,
                    steps=25, p_fixed=None, device=torch.device('cpu'),
                    material=material, substrate=substrate)
                assert result['P'] >= 1.2 * result['D']
                actual = finite_unit(ml.predict_rgb(
                    result['D'], result['H'], result['P'], 0., 'TE', material, substrate), (3,))
                rgb_error = float(np.max(np.abs(actual - result['pred_rgb'])))
                assert rgb_error <= 1e-5, rgb_error
                de = float(delta_e2000(rgb_to_lab(actual), rgb_to_lab(target)))
                assert abs(de - result['de2000']) <= 1e-3
                assert ml.single_inverse_model_status(material, substrate)[0]
                return dict(models=len(paths), spectrum_max_abs_diff=error,
                            applied_geometry_rgb_max_abs_diff=rgb_error,
                            geometry=[result['D'], result['H'], result['P']],
                            delta_e2000=result['de2000'], optimization_steps=25)
            check(f'gradient_and_onnx_parity/{material}/{substrate}', exercise_pair)

    def smart_grid():
        result = ml.smart_grid_search(target, coarse_n=12, top_k=5, fine_steps=5)
        assert result and len(result) == 3
        errors = []
        for _, param, rgb, _, de in result:
            assert param.period_nm >= 1.2 * param.diameter_nm
            actual = ml.predict_rgb(param.diameter_nm, param.height_nm, param.period_nm)
            error = float(np.max(np.abs(np.asarray(rgb) - actual)))
            assert error < 1e-4, error
            assert np.isfinite(de)
            errors.append(error)
        return dict(candidates=len(result), applied_geometry_rgb_max_abs_diff=max(errors))
    check('smart_grid/default_full_search', smart_grid)

    def rl_search():
        import rl_design
        agent = rl_design.get_trained_rl()
        assert agent.q.shape == (rl_design.N_STATES, rl_design.N_ACTIONS)
        assert np.isfinite(agent.q).all() and agent.trained
        d, h, p, color, de = agent.search('#4d66cc', steps=30, restarts=5)
        assert p > d
        actual = finite_unit(ml.predict_rgb(d, h, p), (3,))
        recomputed = float(delta_e2000(rgb_to_lab(actual), rgb_to_lab(np.array([77,102,204])/255)))
        assert abs(de - recomputed) < 1e-6
        return dict(geometry=[d,h,p], color=color, delta_e2000=de,
                    nonpositive_rows=int((agent.q.max(axis=1) <= 0).sum()))
    check('rl/local_qtable_inference', rl_search)

    def analytical():
        count = 0
        for material in ml.MATERIAL_CODES:
            for substrate in ml.SUBSTRATE_CODES:
                for pol in (True, False):
                    for angle in (0., 30.):
                        args = [torch.tensor([v]) for v in (180.,300.,140.,280.,400.)]
                        finite_unit(tm.batch_dual_pillar_spectrum(
                            *args, theta=angle, pol_TE=pol, material=material,
                            substrate=substrate).detach().numpy(), (1,81))
                        count += 1
        return dict(dual_forward_contexts=count)
    check('dual/analytical_forward_matrix', analytical)

    def fp():
        from fp_cavity import fp_cavity_spectrum, fp_dielectric_spectrum
        count = 0
        for fn in (fp_cavity_spectrum, fp_dielectric_spectrum):
            for pol in (True, False):
                for angle in (0.,30.):
                    wl, refl = fn(180., angle_deg=angle, pol_TE=pol)
                    assert len(wl) == len(refl) and len(wl) > 1
                    finite_unit(refl, (len(wl),))
                    count += 1
        return dict(mirror_polarization_angle_contexts=count)
    check('fp/ag_and_dbr_forward', fp)

    def reference():
        from competition.reference_library import load_reference_library
        library = load_reference_library()
        assert library.record_count == 21088
        match = library.lookup(140, 281, 407)
        assert match is not None
        return dict(records=library.record_count, exact_lookup=True)
    check('reference/audited_library_and_exact_lookup', reference)

    def preserved():
        assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                   for path, digest in frozen.items())
        return dict(unchanged_artifacts=len(frozen), training_executed=False)
    check('weights_and_qtable_unchanged', preserved)
    result = dict(schema='offline-feature-runtime-audit-v1', runtime=str(runtime),
                  status='pass' if all(c['status']=='pass' for c in checks) else 'fail',
                  started_at_epoch=started, finished_at_epoch=time.time(),
                  scope='local_inference_and_design_only', physical_accuracy_claim=False,
                  checks=checks, protected_research_modified=False)
    result['software_sha256'] = software
    assert all(hashlib.sha256((runtime/name).read_bytes()).hexdigest() == digest
               for name,digest in software.items()), 'Runtime changed during audit'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(args.output)
    return 0 if result['status']=='pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
