import math
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
    BOUND = 4.0

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
    
class Periodic_Well(Potential):
    """Periodic_Well potential, a periodic potential with multiple wells.
    Input:
        x: Tensor [N, 2]
    Output:
        V: Tensor [N]
    """
    BOUND = math.pi

    def __str__(self):
        return "Periodic Well"
    
    def __repr__(self):
        return "PW"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, 0]
        x2 = x[:, 1]
        sin_2x1 = torch.sin(2 * x1)
        sin_2x2 = torch.sin(2 * x2)
        term_y = torch.sign(sin_2x2) * torch.abs(sin_2x2).pow(1.4)
        return 4.0 * sin_2x1 * term_y
    
if __name__ == "__main__":
    target = Himmelblau() # choose potential
    BOUND = target.BOUND # get bound
    uniform = Uniform(BOUND=BOUND) # set uniform distribution
    N = 5 # number of samples
    x = uniform.samples(N) # uniform distribution
    V = target(x) # evaluate potential
    print("Sampled points:\n", x)
    print("Potential values:\n", V)
    