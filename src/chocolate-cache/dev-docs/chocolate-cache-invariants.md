# Chocolate Cache — Invariants
The finished project MUST comply with the invariants listed here.

-- Experts MUST be spread over all the devices so they can be executed in parallel. For instance, if CUDA0 and CUDA1 devices are available, the cache should spread the experts across both devices so execution of them can be made in parallel.

