"""One consistency-review round: current files -> 3 external reviewers."""
import json, subprocess, sys, pathlib
R = pathlib.Path('../.aris/reviews/poisson-consistency')
n = sys.argv[1]
focus = sys.argv[2] if len(sys.argv) > 2 else ''
alg = subprocess.run(['sed', '-n', '/caption{Adaptive-temperature Boltzmann/,/end{algorithm}/p',
                      '../Paper/main.tex'], capture_output=True, text=True).stdout
files = {'PAPER Algorithm 4 (ground truth)': alg,
         'Clock_Lattice/core/boltzmann.py (trusted original implementation)':
             open('../Clock_Lattice/core/boltzmann.py').read(),
         'Poisson_Inverse/core/boltzmann.py (CURRENT, modified)':
             open('core/boltzmann.py').read(),
         'Poisson_Inverse/train.py (driver, defines fine_fn)': open('train.py').read(),
         'Poisson_Inverse/potential.py (whitened potentials, basis, QT)': open('potential.py').read(),
         'Poisson_Inverse/census.py (offline inference)': open('census.py').read(),
         'Poisson_Inverse/MATH.md Sec 5 (extension-test math)':
             open('MATH.md').read().split('## 5.')[1].split('## 6.')[0],
         'LEDGER (already-adjudicated non-issues -- do NOT re-report)':
             open(R / 'LEDGER.md').read()}
body = "\n\n".join(f"=== {k} ===\n{v}" for k, v in files.items())
prompt = ("You are an adversarial consistency auditor, round " + n + " of an iterative debug. "
"Compare the CURRENT implementation against (a) the paper's Algorithm 4 and (b) the trusted "
"Clock_Lattice original. The current version adds, deliberately (user-approved): persistent "
"compile-once bridge potentials mutated via set_coeffs; 50k chunking; an early-abort (direct "
"ESS<0.05 at 200-step checkpoints); per-stage checkpoints; a stall rule; and a FINE metric: "
"per-stage validation extends the carried pre-images y_tilde ~ nu_k by identity on whitened "
"high modes, logw_fine = logw_low + t_k*(Phi_low(y) - Phi_full([y; xi_h])), gating ONLY the "
"t_k = 1 stage (intermediate stages gate on the low ESS; fine failures at t=1 retrain at t=1 "
"WITHOUT shrinking; low failures below 1 shrink t_k toward t_{k-1} per the paper).\n\n"
"AUDIT: (1) every Algorithm 4 line vs the current code -- U_{k-1} fixed inside the repeat, "
"only t_k mutates, G_k identity re-init per attempt, QT per attempt on the re-formed U_k, "
"pool/batch sourcing, validation update formula, resample+rejuvenate on acceptance only; "
"(2) the set_coeffs bridge refactor: coefficient ORDER vs linear_combination([u, u0], [t, 1-t]), "
"aliasing hazards (u_prev and u_next are the same two objects mutated in place -- find any "
"place where stale coeffs could be read, including QT, langevin, SMC, validation, acceptance "
"rejuvenation ordering); (3) the fine metric: derive logw_fine yourself from the densities and "
"check the implemented formula, chunking, and that the gate logic matches the stated policy; "
"(4) interactions among the added rules (early-abort vs max_skip, stall vs fine-no-shrink, "
"checkpoint timing). Do NOT re-report ledger items. Output STRICTLY: numbered NEW findings, "
"each with severity critical/major/minor, exact quote, why, concrete fix; max 6; if none, "
"output exactly CONSISTENT.\n" + ("ROUND FOCUS (audit this area with maximum depth): " + focus + "\n" if focus else "") + "\n" + body)
json.dump({'prompt': prompt}, open('/tmp/pc_round.json', 'w'))
print('prompt bytes:', len(prompt))
