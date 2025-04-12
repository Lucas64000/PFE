import os

NUM_CPU = os.cpu_count()
NUM_PROC = max(1, min(NUM_CPU - 2, NUM_CPU // 2))