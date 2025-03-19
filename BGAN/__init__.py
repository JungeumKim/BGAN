from .BGAN_critic import Critic
from .BGAN_generator import Generator

# Explicitly define the objects to be exported when "from BGAN import ..." is called
__all__ = ["Critic", "Generator"]
