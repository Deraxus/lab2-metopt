import math, time
import numpy as np
import matplotlib.pyplot as plt

# Разрешённые функции для eval
SAFE_FUNCS = {
    "pi": math.pi, "e": math.e,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "fabs": math.fabs,
}

def make_func(expr: str):
    code = compile(expr, "<user-func>", "eval")
    def f(x):
        return eval(code, {"__builtins__": {}}, {**SAFE_FUNCS, "x": float(x)})
    return f

class PSOptimizerResult:
    def __init__(self, x_min, f_min, iters, elapsed, xs, fs, lower_x, lower_y, L_used):
        self.x_min = x_min
        self.f_min = f_min
        self.iters = iters
        self.elapsed = elapsed
        self.xs = xs
        self.fs = fs
        self.lower_x = lower_x
        self.lower_y = lower_y
        self.L_used = L_used

class PSOptimizer:
    def __init__(self, f, a, b, eps=1e-2, L=None, max_iters=10000):
        if not (a < b):
            raise ValueError("a должен быть меньше b")
        self.f = f
        self.a = float(a)
        self.b = float(b)
        self.eps = float(eps)
        self.max_iters = int(max_iters)
        self.L = float(L) if L is not None else None

    @staticmethod
    def _interval_characteristic(xi, fi, xj, fj, L):
        return (fi + fj - L * (xj - xi)) / 2.0

    @staticmethod
    def _interval_candidate_x(xi, fi, xj, fj, L):
        return 0.5 * (xi + xj) + 0.5 * (fj - fi) / L

    def _update_L(self, xs, fs, Lcur):
        max_slope = 0.0
        for i in range(1, len(xs)):
            dx = abs(xs[i] - xs[i-1])
            if dx == 0:
                continue
            s = abs((fs[i] - fs[i-1]) / dx)
            if s > max_slope:
                max_slope = s
        max_slope = max(max_slope, 1e-9)
        return max(Lcur, 1.1 * max_slope)

    def minimize(self):
        t0 = time.perf_counter()

        xs = [self.a, self.b]
        fs = [self.f(self.a), self.f(self.b)]
        xs, fs = zip(*sorted(zip(xs, fs)))
        xs, fs = list(xs), list(fs)

        L = self.L
        if L is None:
            dx = xs[1] - xs[0]
            L = 1.1 * abs((fs[1] - fs[0]) / dx) if dx != 0 else 1.0
            L = max(L, 1.0)

        best_idx = int(np.argmin(fs))
        f_best = fs[best_idx]

        it = 0
        while it < self.max_iters:
            it += 1

            L = self._update_L(xs, fs, L)

            R = []
            cand_x = []
            for i in range(len(xs) - 1):
                xi, xj = xs[i], xs[i+1]
                fi, fj = fs[i], fs[i+1]
                R.append(self._interval_characteristic(xi, fi, xj, fj, L))
                cand_x.append(self._interval_candidate_x(xi, fi, xj, fj, L))

            R = np.array(R)
            min_R = float(R.min())
            min_i = int(R.argmin())
            x_new = float(cand_x[min_i])

            x_new = max(xs[min_i], min(xs[min_i+1], x_new))

            if f_best - min_R <= self.eps:
                break

            f_new = self.f(x_new)
            pos = np.searchsorted(xs, x_new)
            xs.insert(pos, x_new)
            fs.insert(pos, f_new)

            if f_new < f_best:
                f_best = f_new
                best_idx = pos

        elapsed = time.perf_counter() - t0

        grid = np.linspace(self.a, self.b, 800)
        env = []
        for gx in grid:
            vals = [fs[k] - L * abs(gx - xs[k]) for k in range(len(xs))]
            env.append(max(vals))

        return PSOptimizerResult(xs[best_idx], f_best, it, elapsed, xs, fs, grid, np.array(env), L)

def run_and_save(expr, a, b, eps=0.01, L=None, outfile_prefix="demo"):
    f = make_func(expr)
    opt = PSOptimizer(f, a, b, eps=eps, L=L)
    res = opt.minimize()

    grid = np.linspace(a, b, 1200)
    fvals = np.array([f(x) for x in grid])

    # График функции
    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, fvals, label="Исходная функция f(x)")
    plt.scatter(res.xs, res.fs, s=20, label="Точки вычислений")
    plt.scatter([res.x_min], [res.f_min], s=60, marker="x", label="Найденный минимум")
    plt.title("График функции и вычисленные точки")
    plt.xlabel("Аргумент x")
    plt.ylabel("Значение f(x)")
    plt.legend()
    fig1 = f"{outfile_prefix}_function.png"
    plt.savefig(fig1, bbox_inches="tight", dpi=160)
    plt.close()

    # Нижняя огибающая
    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, fvals, label="Исходная функция")
    plt.plot(res.lower_x, res.lower_y, label="Нижняя оценка")
    plt.title("Нижняя огибающая")
    plt.xlabel("Аргумент x")
    plt.ylabel("Оценка функции")
    plt.legend()
    fig2 = f"{outfile_prefix}_lower_envelope.png"
    plt.savefig(fig2, bbox_inches="tight", dpi=160)
    plt.close()

    return res, fig1, fig2


if __name__ == "__main__":
    print("Запуск тестов...\n")

    tests = [
        ("(x - 2)**2", -5, 5, "test_quad"),
        ("10 + x*x - 10*cos(2*pi*x)", -5.12, 5.12, "test_rastrigin"),
        ("0.3*x + sin(5*x)", -5, 5, "test_sin_trend"),
        ("0.1*x + 0.5*sin(20*x)", -10, 10, "test_saw"),
        ("x + sin(3.14159*x)", -4, 4, "test_prompt"),
    ]

    for expr, a, b, name in tests:
        res, f1, f2 = run_and_save(expr, a, b, outfile_prefix=name)
        print(f"{name}: x* = {res.x_min:.6f}  f(x*) = {res.f_min:.6f}  "
              f"iters = {res.iters}  time = {res.elapsed:.6f} sec")

    print("\ndone")
