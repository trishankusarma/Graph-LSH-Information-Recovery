from dataclasses import dataclass

@dataclass
class Config():
    epochs :int = 5
    max_lap_k :int = 64
    threshold_on_lap_pe :float = 0.95
    hidden_dim :int = 1024
    lap_dim :int = 64
    num_buckets :int = 4
    num_heads :int = 4
    num_spd_bins :int = 16
    dropout :int = 0.1