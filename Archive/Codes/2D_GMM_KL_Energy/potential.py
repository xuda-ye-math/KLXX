import torch

class Potential:
    """The general 2D potential function interface.
    Subclasses must set BOUND: float, the half-width of the domain [-BOUND, BOUND]^2.
    """
    BOUND: float

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'BOUND'):
            raise TypeError(f"{cls.__name__} must define a class attribute BOUND.")
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the potential at given points.
        Input:
            x: Tensor [N, 2]
        Output:
            V: Tensor [N]
        """
        raise NotImplementedError
    
class Uniform(Potential):
    "Uniform distribution with constant potential in [-BOUND, BOUND]^2"
    BOUND = 0.0
    
    def __init__(self, BOUND: float):
        self.BOUND = BOUND

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    def samples(self, N: int) -> torch.Tensor:
        return 2 * self.BOUND * torch.rand(N, 2) - self.BOUND
    
class GMM(Potential):
    "Gaussian mixture distribution determined by (w_i, μ_i, σ_i)"
    BOUND = 0.0 # bound for GMM is invalid here.
    
    # load from prior_{name}.pt
    def load(self, name: str):
        ckpt = torch.load(f"prior_{name}.pt", weights_only=False)
        self.w_list = ckpt['w_list']
        self.μ_list = ckpt['μ_list']
        self.σ_list = ckpt['σ_list']

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # use logexpsum
        # compute the unormalized density (w_list not required to sum as 1)
        # x: [N, 2], μ_list: [K, 2], σ_list: [K], w_list: [K]
        # density(x) ∝ Σ_i w_i / σ_i^2 * exp(-||x - μ_i||^2 / (2 σ_i^2))
        # V(x) = -log density(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
        μ1 = self.μ_list[:, 0]
        μ2 = self.μ_list[:, 1]
        σ  = self.σ_list
        w  = self.w_list
        sq_dist = (x1.unsqueeze(1) - μ1.unsqueeze(0)).square() \
                + (x2.unsqueeze(1) - μ2.unsqueeze(0)).square()  # [N, K]
        log_terms = torch.log(w).unsqueeze(0) \
                  - 2 * torch.log(σ).unsqueeze(0) \
                  - sq_dist / (2 * σ.square().unsqueeze(0))      # [N, K]
        return -torch.logsumexp(log_terms, dim=1)
    
    def samples(self, N: int) -> torch.Tensor:
        probs = self.w_list / self.w_list.sum()
        idx = torch.multinomial(probs, N, replacement=True)  # [N]
        μ = self.μ_list[idx]                                  # [N, 2]
        σ = self.σ_list[idx].unsqueeze(1)                     # [N, 1]
        return μ + σ * torch.randn(N, 2)

class Himmelblau(Potential):
    """Himmelblau's function, a potential function with 4 wells.
    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    BOUND = 6.0

    def __str__(self):
        return "Himmelblau"
    
    def __repr__(self):
        return "HB"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1.square() + x2 - 11).square()
        term2 = (x1 + x2.square() - 7).square()
        return 0.2 * (term1 + term2)
    
class Annulus(Potential):
    """Annulus potential, a ring-shaped potential with a circular well.
    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    BOUND = 3.0

    def __str__(self):
        return "Annulus"
    
    def __repr__(self):
        return "AN"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r2 = x.square().sum(dim=1)
        inner_term = r2.pow(3) - 8 * r2.pow(2) + 16 * r2 + 1
        return 10.0 * (inner_term.pow(1 / 3) - 1)
    
class Three_Well(Potential):
    """Three_Well potential, a potential with three wells.
    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    BOUND = 2.0

    def __str__(self):
        return "Three Well"
    
    def __repr__(self):
        return "TW"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        term1 = (x1.square() - 1).square()
        term2 = (x2.square() - 1).square()
        term3 = torch.sin(x1 + 2 * x2)
        return 3.0 * (term1 + term2 + term3)
    
class Rosenbrock(Potential):
    """Rosenbrock potential, a banana-shaped thin distribution
    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    BOUND = 4.0

    def __str__(self):
        return "Rosenbrock"
    
    def __repr__(self):
        return "RB"
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        return 2 * (x1 - 0.8) ** 2 + 200 * (x2 - 0.6 * x1 ** 2 + 3.5) ** 2

if __name__ == "__main__":
    target = Himmelblau() # choose potential
    BOUND = target.BOUND # get bound
    uniform = Uniform(BOUND=BOUND) # set uniform distribution
    N = 5 # number of samples
    x = uniform.samples(N) # uniform distribution
    V = target(x) # evaluate potential
    print("Sampled points:\n", x)
    print("Potential values:\n", V)
    