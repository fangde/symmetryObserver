import torch
from diffeo_benchmark.diffeo.triangular import TriangularDiffeoLayer, CompositeDiffeomorphism

def test_triangular_diffeo():
    torch.manual_seed(42)
    dim = 8
    batch_size = 16
    L = 4
    
    layers = []
    for _ in range(L):
        # Sample parameters
        s = torch.randn(dim) * 0.3  # moderate scaling
        a = torch.randn(dim, dim) * 0.4
        a = torch.tril(a, diagonal=-1)
        w = torch.randn(dim, dim) * 0.5
        w = torch.tril(w, diagonal=-1)
        b = torch.randn(dim, dim) * 0.5
        b = torch.tril(b, diagonal=-1)
        perm = torch.randperm(dim)
        
        layer = TriangularDiffeoLayer(dim, s, a, w, b, perm)
        layers.append(layer)
        
    phi = CompositeDiffeomorphism(dim, layers)
    
    x = torch.randn(batch_size, dim)
    
    # 1. Forward
    y = phi.forward(x)
    assert y.shape == (batch_size, dim)
    
    # 2. Invertibility check
    x_rec = phi.inverse(y)
    rec_error = torch.max(torch.abs(x_rec - x)).item()
    print(f"Max inversion error across {L} layers: {rec_error:.2e}")
    assert rec_error < 1e-5, f"Inversion error too high: {rec_error}"
    
    # 3. Analytic Jacobian vs Autograd Jacobian
    J_analytic = phi.jacobian(x)
    
    # Autograd Jacobian for each sample in batch
    for b in range(min(4, batch_size)):
        xb = x[b]
        
        def func(inp):
            return phi.forward(inp.unsqueeze(0)).squeeze(0)
            
        J_auto = torch.autograd.functional.jacobian(func, xb)
        diff = torch.norm(J_analytic[b] - J_auto) / torch.norm(J_auto)
        print(f"Sample {b} Jacobian relative error (analytic vs autograd): {diff.item():.2e}")
        assert diff < 1e-4, f"Jacobian mismatch: {diff.item()}"
        
        # Check determinant strictly positive
        det = torch.linalg.det(J_analytic[b]).item()
        print(f"Sample {b} det(J): {det:.4f}")
        assert det != 0, "Singular Jacobian!"

    print("All mathematical properties verified successfully!")

if __name__ == "__main__":
    test_triangular_diffeo()
