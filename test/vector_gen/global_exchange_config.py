from dataclasses import dataclass, field
from math import log2


@dataclass(frozen=True)
class GlobalExchangeConfig:
    V: int = 128
    F: int = 8
    V_nbits: int = field(init=False)

    def __post_init__(self):
        assert log2(self.V).is_integer()
        V_nbits = int(log2(self.V))
        object.__setattr__(self, 'V_nbits', V_nbits)


global_exchange_config = GlobalExchangeConfig()
