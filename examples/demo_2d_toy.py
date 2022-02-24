import max_inner_ellipsoid
import numpy as np


def demo2D():
    pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
    dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
    max_iterations = 30
    convergence_tol = 0.001
    seed_point = np.array([0.1, 0.3])

    def plot(P_val, q_val, r_val):
        max_inner_ellipsoid.draw_ellipsoid(P_val, q_val, r_val, pts,
                                           np.empty((0, 2)))

    P, q, r = dut.search(seed_point,
                         max_iterations,
                         convergence_tol,
                         delta=np.inf,
                         callback=plot)


if __name__ == "__main__":
    demo2D()
