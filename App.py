"""
1D FDM Solver — Streamlit Web App
u'' + f(x)*u' + g(x)*u = h(x)  on [a,b]
Dirichlet or Neumann BC on each boundary
"""

import re, time, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, cg, gmres

matplotlib.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX 변환
# ══════════════════════════════════════════════════════════════════════════════
def to_latex(s: str) -> str:
    s = s.strip()
    s = re.sub(r'log10\(', r'\\log_{10}(', s)
    s = re.sub(r'log2\(',  r'\\log_{2}(', s)
    s = re.sub(r'(\w+)\*\*(\w+)', r'\1^{\2}', s)
    s = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', s)
    s = re.sub(r'([a-zA-Z])\*([a-zA-Z0-9])', r'\1\2', s)
    for fn in ["sin","cos","tan","sinh","cosh","tanh",
               "exp","log","sqrt","abs"]:
        s = re.sub(rf'\b{fn}\b', rf'\\{fn}', s)
    return s

def make_latex_eq(f_str, g_str, h_str) -> str:
    fl = to_latex(f_str) if f_str.strip() else "0"
    gl = to_latex(g_str) if g_str.strip() else "0"
    hl = to_latex(h_str) if h_str.strip() else "0"
    conv = "" if fl=="0" else ("u'" if fl=="1" else rf"\left({fl}\right)u'")
    reac = "" if gl=="0" else (" + u" if gl=="1" else rf" + \left({gl}\right)u")
    sep  = " + " if conv else ""
    return rf"u'' {sep}{conv}{reac} = {hl}"


# ══════════════════════════════════════════════════════════════════════════════
#  FDM 솔버
# ══════════════════════════════════════════════════════════════════════════════
_ENV_BASE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi, "e": np.e, "np": np,
}

def fdm_solve(f_expr, g_expr, h_expr,
              x_a, x_b, N,
              bc_L, val_L, bc_R, val_R,
              solver="spsolve", maxiter=1000, rtol=1e-10, restart=20):
    t0 = time.perf_counter()
    dx = (x_b - x_a) / N
    x  = x_a + (np.arange(N) + 0.5) * dx
    env = {"x": x, **_ENV_BASE}

    fi = eval(f_expr, env) * np.ones(N)
    gi = eval(g_expr, env) * np.ones(N)
    hi = eval(h_expr, env) * np.ones(N)

    inv_dx2 = 1.0/dx**2
    inv_2dx = 0.5/dx
    a = inv_dx2 - fi*inv_2dx
    b = -2.0*inv_dx2 + gi
    c = inv_dx2 + fi*inv_2dx
    rhs = hi.copy()

    if bc_L == "Dirichlet":
        b[0] -= a[0];    rhs[0]  -= 2.0*a[0]*val_L
    else:
        b[0] += a[0];    rhs[0]  += a[0]*dx*val_L

    if bc_R == "Dirichlet":
        b[-1] -= c[-1];  rhs[-1] -= 2.0*c[-1]*val_R
    else:
        b[-1] += c[-1];  rhs[-1] -= c[-1]*dx*val_R

    A_csr = diags([a[1:], b, c[:-1]], offsets=[-1,0,1],
                  shape=(N,N), format="csr")
    t1 = time.perf_counter()

    iters = None; converged = None
    if solver == "spsolve":
        u = spsolve(A_csr, rhs)
    elif solver == "cg":
        class _C:
            def __init__(self): self.n=0
            def __call__(self,_): self.n+=1
        cnt=_C()
        u, info = cg(A_csr, rhs, maxiter=maxiter, rtol=rtol,
                     atol=1e-12, callback=cnt)
        iters=cnt.n; converged=(info==0)
    elif solver == "gmres":
        class _C:
            def __init__(self): self.n=0
            def __call__(self,_): self.n+=1
        cnt=_C()
        u, info = gmres(A_csr, rhs, restart=restart,
                        maxiter=maxiter, rtol=rtol, atol=1e-12,
                        callback=cnt, callback_type="legacy")
        iters=cnt.n; converged=(info==0)
    else:
        u = spsolve(A_csr, rhs)

    t2 = time.perf_counter()
    timing = {"assemble": t1-t0, "solve": t2-t1,
              "total": t2-t0, "iters": iters, "converged": converged}
    return x, u, timing


# ══════════════════════════════════════════════════════════════════════════════
#  Export helper
# ══════════════════════════════════════════════════════════════════════════════
def make_export(x_src, solutions, x_a, x_b, N, dx,
                bc_L, val_L, bc_R, val_R,
                x_mode, npts, fmt):
    if x_mode == "face":
        x_export = x_a + np.arange(N+1) * dx
    else:
        x_export = np.linspace(x_a, x_b, npts)

    x_aug = np.concatenate([[x_a], x_src, [x_b]])
    cols  = [x_export]
    names = list(solutions.keys())
    for s in names:
        u_sol = solutions[s]
        u_L = val_L if bc_L=="Dirichlet" else u_sol[0]-dx/2*val_L
        u_R = val_R if bc_R=="Dirichlet" else u_sol[-1]+dx/2*val_R
        u_aug = np.concatenate([[u_L], u_sol, [u_R]])
        cols.append(np.interp(x_export, x_aug, u_aug))

    out = np.column_stack(cols)
    delim = "," if fmt=="csv" else "  "
    header = ("x," + ",".join(names)) if fmt=="csv" else \
             ("x" + "".join(f"  {s:>18s}" for s in names))
    buf = io.StringIO()
    buf.write("# " + header + "\n")
    np.savetxt(buf, out, delimiter=delim, fmt="%.10e")
    return buf.getvalue(), x_export.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
#  Streamlit 앱
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="1D FDM Solver", layout="wide",
                   page_icon="📐")

st.title("📐 1D FDM Solver")
st.caption("u'' + f(x)·u' + g(x)·u = h(x)  —  Ghost-cell FVM, "
           "Central difference, Dirichlet / Neumann BC")

# ── 레이아웃: 왼쪽 설정 | 오른쪽 결과 ────────────────────────────────────────
col_L, col_R = st.columns([1, 2], gap="large")

with col_L:
    # ── ODE 입력 ──────────────────────────────────────────────────────────────
    st.subheader("ODE 설정")
    c1, c2, c3 = st.columns(3)
    f_str = c1.text_input("f(x) =", "x**2",
        help="예) x**2, sin(x), exp(x), log(x)=ln, log10(x)")
    g_str = c2.text_input("g(x) =", "3*x")
    h_str = c3.text_input("h(x) =", "4*x")

    # 수식 미리보기
    try:
        latex_eq = make_latex_eq(f_str, g_str, h_str)
        st.latex(latex_eq)
    except Exception:
        st.info(f"u'' + ({f_str})u' + ({g_str})u = {h_str}")

    st.divider()

    # ── 도메인 ────────────────────────────────────────────────────────────────
    st.subheader("도메인 & 경계 조건")
    da, db = st.columns(2)
    x_a = da.number_input("x 시작", value=0.0, step=0.1, format="%.4f")
    x_b = db.number_input("x 끝",   value=1.0, step=0.1, format="%.4f")

    # ── BC ────────────────────────────────────────────────────────────────────
    bc_types = ["Dirichlet  u(x) =", "Neumann  u'(x) ="]
    bL1, bL2 = st.columns(2)
    bc_L_label = bL1.selectbox("왼쪽 BC", bc_types, key="bcL")
    val_L = bL2.number_input("값  (왼쪽)", value=1.0, format="%.4f",
                              key="valL")
    bc_L = "Dirichlet" if "Dirichlet" in bc_L_label else "Neumann"

    bR1, bR2 = st.columns(2)
    bc_R_label = bR1.selectbox("오른쪽 BC", bc_types, key="bcR")
    val_R = bR2.number_input("값  (오른쪽)", value=2.0, format="%.4f",
                              key="valR")
    bc_R = "Dirichlet" if "Dirichlet" in bc_R_label else "Neumann"

    lbl_L = f"u({x_a})={val_L}" if bc_L=="Dirichlet" else f"u'({x_a})={val_L}"
    lbl_R = f"u({x_b})={val_R}" if bc_R=="Dirichlet" else f"u'({x_b})={val_R}"
    st.caption(f"왼쪽: **{lbl_L}**   |   오른쪽: **{lbl_R}**")

    st.divider()

    # ── 수치 파라미터 ─────────────────────────────────────────────────────────
    st.subheader("수치 파라미터")
    N = st.number_input("셀 수 N", min_value=4, max_value=5000,
                        value=10, step=1)
    N = int(N)

    SOLVER_OPTS = {
        "Direct (spsolve)":   "spsolve",
        "Iterative CG":       "cg",
        "Iterative GMRES":    "gmres",
    }
    solver_label = st.selectbox("Solver", list(SOLVER_OPTS.keys()))
    sel_solver   = SOLVER_OPTS[solver_label]
    compare_all  = st.checkbox("3개 Solver 모두 비교")

    if sel_solver in ("cg","gmres") or compare_all:
        with st.expander("Iterative Solver 설정", expanded=True):
            maxiter = st.number_input("Max iter", 1, 100000, 1000)
            rtol    = st.text_input("rtol", "1e-10")
            try: rtol = float(rtol)
            except: rtol = 1e-10
            if sel_solver == "gmres" or compare_all:
                restart = st.number_input(
                    "Restart m (GMRES)", 1, 10000, 20,
                    help="GMRES(m) restart 길이. 20~50 권장.")
                restart = int(restart)
            else:
                restart = 20
    else:
        maxiter = 1000; rtol = 1e-10; restart = 20

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.subheader("데이터 Export")
    x_mode = st.radio("x 기준",
                      ["Face 위치 (N+1 점)", "균등 분할"],
                      horizontal=True)
    x_mode_key = "face" if "Face" in x_mode else "uniform"
    npts = N+1
    if x_mode_key == "uniform":
        npts = int(st.number_input("점 수", 2, 100000, 100))
    exp_fmt = st.radio("포맷", ["공백 구분 (.txt)", "쉼표 구분 (.csv)"],
                       horizontal=True)
    fmt_key = "csv" if "csv" in exp_fmt else "txt"

    # ── Run 버튼 ──────────────────────────────────────────────────────────────
    run_btn = st.button("▶  Run", type="primary", use_container_width=True)

# ── 오른쪽: 결과 ──────────────────────────────────────────────────────────────
with col_R:
    if x_b <= x_a:
        st.error("x 끝 > x 시작 이어야 합니다.")
        st.stop()

    solvers = (["spsolve","cg","gmres"]
               if compare_all else [sel_solver])

    COLORS = {"spsolve":"#1560A8","cg":"#C0392B","gmres":"#1D9E75"}
    LABELS = {"spsolve":"Direct (spsolve)",
              "cg":"Iterative CG","gmres":"Iterative GMRES"}

    if run_btn:
        solutions = {}; timings = {}
        dx = (x_b - x_a) / N
        x_src = None

        prog = st.progress(0, text="계산 중…")
        for idx, s in enumerate(solvers):
            try:
                xc, u, tm = fdm_solve(
                    f_str, g_str, h_str, x_a, x_b, N,
                    bc_L, val_L, bc_R, val_R,
                    solver=s, maxiter=maxiter, rtol=rtol, restart=restart)
                solutions[s] = u
                timings[s]   = tm
                x_src        = xc
                prog.progress((idx+1)/len(solvers),
                              text=f"{s} 완료")
            except Exception as e:
                st.error(f"오류 ({s}): {e}")
                prog.empty()
                st.stop()
        prog.empty()

        # 세션 상태에 저장 (Export 버튼용)
        st.session_state["solutions"] = solutions
        st.session_state["x_src"]     = x_src
        st.session_state["x_a"]       = x_a
        st.session_state["x_b"]       = x_b
        st.session_state["N"]         = N
        st.session_state["dx"]        = dx
        st.session_state["bc_L"]      = bc_L
        st.session_state["val_L"]     = val_L
        st.session_state["bc_R"]      = bc_R
        st.session_state["val_R"]     = val_R

        # ── 그래프 ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        for s, u in solutions.items():
            ax.plot(xc, u, "o-", color=COLORS[s], lw=2, ms=5,
                    label=LABELS[s])
        if bc_L == "Dirichlet":
            ax.plot(x_a, val_L, "^k", ms=9, zorder=6)
        if bc_R == "Dirichlet":
            ax.plot(x_b, val_R, "^k", ms=9, zorder=6,
                    label=f"BC: {lbl_L},  {lbl_R}")
        for xf in np.linspace(x_a, x_b, N+1):
            ax.axvline(xf, color="#EEEEEE", lw=0.7)
        ax.set_xlabel("x", fontsize=12); ax.set_ylabel("u(x)", fontsize=12)
        ax.set_xlim(x_a - dx*0.3, x_b + dx*0.3)
        ax.legend(fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False); ax.tick_params(bottom=False, left=False)
        try:
            eq_title = f"$u'' + ({f_str})u' + ({g_str})u = {h_str}$"
            ax.set_title(eq_title + f"\nN={N},  [{x_a},{x_b}],  "
                         f"{lbl_L},  {lbl_R}", fontsize=10)
        except Exception:
            pass
        st.pyplot(fig, use_container_width=True)

        # ── 계산 시간 / 수치 요약 ─────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**수치 요약**")
            rows = []
            for s, u in solutions.items():
                rows.append({"solver": s,
                             "min": f"{u.min():.6f}",
                             "max": f"{u.max():.6f}"})
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**계산 시간**")
            t_rows = []
            for s, tm in timings.items():
                istr = ""
                if tm["iters"] is not None:
                    icon = "✓" if tm["converged"] else "✗"
                    istr = f"{icon} {tm['iters']} iters"
                t_rows.append({
                    "solver": s,
                    "조립 (ms)": f"{tm['assemble']*1e3:.3f}",
                    "풀기 (ms)": f"{tm['solve']*1e3:.3f}",
                    "합계 (ms)": f"{tm['total']*1e3:.3f}",
                    "수렴": istr,
                })
            st.dataframe(t_rows, use_container_width=True, hide_index=True)

    # ── Export 버튼 (세션에 결과 있을 때만) ──────────────────────────────────
    if "solutions" in st.session_state:
        st.divider()
        ss = st.session_state
        ext  = ".csv" if fmt_key=="csv" else ".txt"
        data_str, n_out = make_export(
            ss["x_src"], ss["solutions"],
            ss["x_a"], ss["x_b"], ss["N"], ss["dx"],
            ss["bc_L"], ss["val_L"], ss["bc_R"], ss["val_R"],
            x_mode_key, npts, fmt_key)
        st.download_button(
            label=f"💾  Export ({n_out}pts, linear 보간){ext}",
            data=data_str,
            file_name=f"fdm_solution{ext}",
            mime="text/plain",
            use_container_width=True,
        )