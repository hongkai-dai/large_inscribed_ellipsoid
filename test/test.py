import max_inner_ellipsoid

import numpy as np
import unittest

class TestSearchLargeEllipsoid(unittest.TestCase):
    def test_constructor(self):
        # 2D case
        pts = np.array([[-1., -2.],
                        [2, 3.],
                        [0, 4.],
                        [1, 2.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        self.assertEqual(dut.dim, 2)
        np.testing.assert_array_less(dut.C @ np.mean(pts, axis=0), dut.d)
        for i in range(pts.shape[0]):
            np.testing.assert_array_less(dut.C @ pts[i], dut.d + 1E-6)

    def test_find_initial_ellipsoid(self):
        # 2D case
        pts = np.array([[-1, -2],
                        [-1, 2.],
                        [1., -2],
                        [1, 2.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        P0, q0, r0 = dut._find_initial_ellipsoid(np.array([0., 0.]))
        # This returned ellipsoid should be x[0]**2 + x[1]**2/4<=1
        np.testing.assert_allclose(P0, np.diag([1., 0.25]) * r0, atol=1E-6)
        np.testing.assert_allclose(q0, np.array([0., 0.]) * r0, atol=1E-6)

        pts = np.array([[-1, -2],
                        [-1, 2.],
                        [1., -2],
                        [1, 2.],
                        [0, 1.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        P0, q0, r0 = dut._find_initial_ellipsoid(np.array([0., 0.]))
        # This returned ellipsoid should be x[0]**2 + (x[1]-0.5)**2 / 1.5**2 <= 1
        np.testing.assert_allclose(P0 / r0, np.diag([1., 1./1.5**2]) / (8.0/9.0), atol=1E-6)
        np.testing.assert_allclose(q0 / r0, np.array([0, 0.5 / (1.5**2)]) / (8.0/9.0), atol=1E-6)


class TestFindEllipsoid(unittest.TestCase):
    def test_feasible(self):
        outside_pts = np.array(
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.2, 0.4]])
        inside_pts = np.array([[0.4, 0.5], [0.5, 0.6], [0.7, 0.2]])

        A, b, hull = max_inner_ellipsoid.get_hull(outside_pts)

        P, q, r, lambda_val = max_inner_ellipsoid.find_ellipsoid(
            outside_pts, inside_pts, A, b)

        self.assertIsNotNone(P)
        # check inside_pts are actually inside the ellipsoid.
        np.testing.assert_array_less(
            np.sum(inside_pts.T * (P @ inside_pts.T), axis=0) + 
            2 * inside_pts @ q, (r + 1e-5) * np.ones(inside_pts.shape[0]))
        # check outside_pts are actually outside the ellipsoid
        np.testing.assert_array_less(
            (r - 1e-5) * np.ones(outside_pts.shape[0]),
            np.sum(outside_pts.T * (P @ outside_pts.T), axis=0) + 
            2 * outside_pts @ q, )
        # check that the ellipsoid is actually within the convex hull.
        np.testing.assert_array_less(0, lambda_val)
        for i in range(A.shape[0]):
            Qi = np.zeros((3, 3))
            Qi[:2, :2] = P
            Qi[:2, 2] = q - lambda_val[i] * A[i, :]/2
            Qi[:2, 2] = q - lambda_val[i] * A[i, :]/2
            Qi[2, 2] = lambda_val[i] * b[i] - r
            eig_value, _ = np.linalg.eig(Qi)
            np.testing.assert_array_less(0, eig_value)

    def test_infeasible(self):
        outside_pts = np.array(
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.], [0.5, 0.5]])
        inside_pts = np.array([[0.3, 0.3], [0.7, 0.7]])

        A, b, hull = max_inner_ellipsoid.get_hull(outside_pts)
        P, q, r, lambda_val = max_inner_ellipsoid.find_ellipsoid(
            outside_pts, inside_pts, A, b)
        self.assertIsNone(P)
        self.assertIsNone(q)
        self.assertIsNone(r)
        self.assertIsNone(lambda_val)


if __name__ == "__main__":
    unittest.main()
