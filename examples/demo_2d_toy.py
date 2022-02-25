import max_inner_ellipsoid
import numpy as np
import matplotlib.pyplot as plt


def demo2D():
    pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
    dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
    max_iterations = 30
    convergence_tol = 1E-6
    seed_point = np.array([0.1, 0.3])

    iter_count = 0

    def plot(P_val, q_val, r_val):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        max_inner_ellipsoid.draw_ellipsoid(ax, P_val, q_val, r_val, pts,
                                           np.empty((0, 2)))
        nonlocal iter_count
        ax.set_title(f"iter={iter_count}")
        fig.savefig(f"iter{iter_count:02d}.png", format="png")
        iter_count += 1

    P, q, r = dut.search(seed_point,
                         max_iterations,
                         convergence_tol,
                         delta=np.inf,
                         callback=plot)


if __name__ == "__main__":
    demo2D()
