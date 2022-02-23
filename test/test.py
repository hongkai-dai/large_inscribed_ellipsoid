import max_inner_ellipsoid

import numpy as np
import cvxpy as cp
import unittest


def check_inscribed_ellipsoid(P, q, r, vertices, C, d, tol):
    # Check if the ellipsoid {x | xᵀPx+2qᵀx≤r} is contained within Cx<=d and
    # doesn't contain any row of vertices.
    np.testing.assert_array_less(
        r - tol,
        np.sum(vertices.T * (P @ (vertices.T)), axis=0) + 2 * q @ (vertices.T))
    # Now write the ellipsoid as {Eu+f | |u|<=1}.
    # The ellipsoid xᵀPx+2qᵀx≤r can be written as
    # |(P_chol*x + P_chol⁻ᵀ*q) / sqrt(r + qᵀP⁻¹q)| <= 1
    # If we compare the terms, we have
    # E⁻¹ = P_chol / sqrt(r+qᵀP⁻¹q)
    # E⁻¹f = -P_chol⁻ᵀ*q/sqrt(r + qᵀP⁻¹q)
    radius = np.sqrt(r + q.dot(np.linalg.solve(P, q)))
    # P_chol.T @ P_chol = P
    P_chol = np.linalg.cholesky(P).T
    E = np.linalg.inv(P_chol / radius)
    f = -E @ (np.linalg.solve(P_chol.T, q) / radius)
    # The ellipsoid {Eu+f | |u|<=1} is within the halfspace
    # {x|cᵢᵀx <= dᵢ} iff |cᵢᵀE| + cᵢᵀf ≤ dᵢ
    np.testing.assert_array_less(
        np.linalg.norm(C @ E, axis=1) + C @ f, d + tol)


class TestSearchLargeEllipsoid(unittest.TestCase):
    def test_constructor(self):
        # 2D case
        pts = np.array([[-1., -2.], [2, 3.], [0, 4.], [1, 2.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        self.assertEqual(dut.dim, 2)
        np.testing.assert_array_less(dut.C @ np.mean(pts, axis=0), dut.d)
        for i in range(pts.shape[0]):
            np.testing.assert_array_less(dut.C @ pts[i], dut.d + 1E-6)

    def test_find_initial_ellipsoid(self):
        # 2D case
        pts = np.array([[-1, -2], [-1, 2.], [1., -2], [1, 2.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        P0, q0, r0 = dut._find_initial_ellipsoid(np.array([0., 0.]))
        # This returned ellipsoid should be x[0]**2 + x[1]**2/4<=1, touching
        # the boundary of the convex hull of pts.
        np.testing.assert_allclose(P0, np.diag([1., 0.25]) * r0, atol=1E-6)
        np.testing.assert_allclose(q0, np.array([0., 0.]) * r0, atol=1E-6)

        pts = np.array([[-1, -2], [-1, 2.], [1., -2], [1, 2.], [0, 1.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        P0, q0, r0 = dut._find_initial_ellipsoid(np.array([0., 0.]))
        # This returned ellipsoid should be
        # x[0]**2 + (x[1]-0.5)**2 / 1.5**2 <= 1, it touches [0, 1] in pts.
        np.testing.assert_allclose(P0 / r0,
                                   np.diag([1., 1. / 1.5**2]) / (8.0 / 9.0),
                                   atol=1E-6)
        np.testing.assert_allclose(q0 / r0,
                                   np.array([0, 0.5 / (1.5**2)]) / (8.0 / 9.0),
                                   atol=1E-6)
        # Now check if the returned P0, q0, r0 satsify our constraints for
        # search_around.
        constraints = []
        P = cp.Variable((2, 2), symmetric=True)
        q = cp.Variable(2)
        r = cp.Variable()
        for i in range(pts.shape[0]):
            constraints.append(
                pts[i, :] @ (P @ pts[i, :]) + 2 * q @ pts[i, :] >= r)
        for i in range(dut.C.shape[0]):
            face_constraints, _ = \
                max_inner_ellipsoid.add_ellipsoid_inside_halfspace(
                    P, q, r, dut.C[i], dut.d[i])
            constraints.extend(face_constraints)
        constraints.append(P == P0)
        constraints.append(q == q0)
        constraints.append(r == r0)
        prob = cp.Problem(cp.Maximize(0), constraints)
        prob.solve()
        self.assertEqual(prob.status, "optimal")

    def test_search_around(self):
        # A 2D example
        pts = np.array([[-1, -2.], [1., -2.], [-1., 2.], [1., 2.], [0, 1.]])
        dut = max_inner_ellipsoid.SearchLargeEllipsoid(pts)
        P0, q0, r0 = dut._find_initial_ellipsoid(np.array([0., 0.]))
        # Use a trust region with different radius
        for delta in (0.1, 1, 2, np.inf):
            Pbar1, qbar1, rbar1 = dut._search_around(P0, q0, r0, delta)
            # Make sure that the new ellipsoid contains z1 as the center for
            # the old ellipsoid.
            z1 = -np.linalg.solve(P0, q0)
            self.assertLessEqual(z1.dot(Pbar1 @ z1) + 2 * qbar1.dot(z1), rbar1)
            # When delta=inf, the ellipsoid is really badly scaled with bad
            # numerics, hence we use a very large tolerance.
            tol = 1E-5
            check_inscribed_ellipsoid(Pbar1, qbar1, rbar1, pts, dut.C, dut.d,
                                      tol)


class TestFindEllipsoid(unittest.TestCase):
    def test_feasible(self):
        outside_pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                [0.2, 0.4]])
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
            2 * outside_pts @ q,
        )
        # check that the ellipsoid is actually within the convex hull.
        np.testing.assert_array_less(0, lambda_val)
        for i in range(A.shape[0]):
            Qi = np.zeros((3, 3))
            Qi[:2, :2] = P
            Qi[:2, 2] = q - lambda_val[i] * A[i, :] / 2
            Qi[:2, 2] = q - lambda_val[i] * A[i, :] / 2
            Qi[2, 2] = lambda_val[i] * b[i] - r
            eig_value, _ = np.linalg.eig(Qi)
            np.testing.assert_array_less(0, eig_value)

    def test_infeasible(self):
        outside_pts = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                [0.5, 0.5]])
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
