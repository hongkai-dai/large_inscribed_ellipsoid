import numpy as np
import torch
import unittest


class EllipsoidVolumeGradientTest(unittest.TestCase):
    """
    The volume of an ellipsoid xᵀPx+2qᵀx≤r is
    power(r+qᵀP⁻¹q, n/2) / sqrt(det(P)) * power(π, n/2)/Γ(n+1/2),
    where power(π, n/2)/Γ(n+1/2) is a constant. To maximize the ellipsoid
    volume, I can maximize its logarithm (modulo a constant) as
    V(P, q, r) = n*log(r+qᵀP⁻¹q) - log(det(P)).
    I want to make sure the gradient of this V(P, q, r) is computed correctly
    as
    n [r qᵀ]⁻¹ - (n+1)P⁻¹
      [q -P]
    """
    def volume_gradient_tester(self, P: torch.Tensor, q: torch.Tensor,
                               r: torch.Tensor):
        n = P.shape[0]
        V = n * torch.log(r + q.dot(torch.linalg.solve(P, q))) - torch.log(
            torch.linalg.det(P))
        V.backward()
        P_grad_expected = P.grad.clone()
        q_grad_expected = q.grad.clone()
        r_grad_expected = r.grad.clone()
        # We call the matrix [r qᵀ] as X
        #                    [q -P]
        X = torch.empty((n + 1, n + 1), dtype=P.dtype)
        X[0, 0] = r
        X[0, 1:] = q.T
        X[1:, 0] = q
        X[1:, 1:] = -P
        X_inv = torch.linalg.inv(X)
        P_inv = torch.linalg.inv(P)
        r_grad = n * X_inv[0, 0]
        q_grad = n * X_inv[0, 1:] * 2
        P_grad = n * -X_inv[1:, 1:] - (n + 1) * P_inv
        self.assertAlmostEqual(r_grad.item(), r_grad_expected.item())
        np.testing.assert_allclose(q_grad.detach().numpy(),
                                   q_grad_expected.detach().numpy())
        np.testing.assert_allclose(P_grad.detach().numpy(),
                                   P_grad_expected.detach().numpy())

    def test_2d_ellipsoid(self):
        dtype = torch.float64
        P = torch.tensor([[1, 3], [3, 25]], dtype=dtype)
        P.requires_grad = True
        q = torch.tensor([2, 5], dtype=dtype)
        q.requires_grad = True
        r = torch.tensor(10, dtype=dtype)
        r.requires_grad = True

        self.volume_gradient_tester(P, q, r)

    def test_3d_ellipsoid(self):
        dtype = torch.float64
        P = torch.tensor([[9, 5, 3], [5, 20, 2], [3, 2, 40]], dtype=dtype)
        P.requires_grad = True
        q = torch.tensor([2, 5, 3], dtype=dtype)
        q.requires_grad = True
        r = torch.tensor(20, dtype=dtype)
        r.requires_grad = True
        self.volume_gradient_tester(P, q, r)


if __name__ == "__main__":
    unittest.main()
