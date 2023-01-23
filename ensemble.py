from src.relays import EnsembleRelay

if __name__ == "__main__":
    EnsembleRelay.with_hydra(root="conf", clear_cache=True)
