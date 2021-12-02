"""
Test a simple example in 3D.
"""
import max_inner_ellipsoid
import numpy as np

def demo3D():
    pts = np.array([[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [1., -1., -1.], [-1., 1., 1.], [-1., 1., -1.], [-1., -1., 1.], [-1., -1., -1], [0., 0., 1.], [0., 0., -1.], [0., 1., 0], [0., -1., 0.], [1., 0., 0.], [-1., 0., 0.]])
    max_iterations = 10
    volume_increase_tol = 0.001
    dut = max_inner_ellipsoid.FindLargeEllipsoid(pts)
    P, q, r, outside_pts, inside_pts = dut.search(max_iterations, volume_increase_tol)
    max_inner_ellipsoid.draw_ellipsoid(P, q, r, outside_pts, inside_pts)

if __name__ == "__main__":
    demo3D()
