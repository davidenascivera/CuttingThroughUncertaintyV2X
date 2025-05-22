from main import Main
from scipy.stats import ttest_ind
import numpy as np
import logging
import sys

# Reset logging
for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler("log_experiments.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ]
)

# Show only logs from this script
class ScriptOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.name == "__main__"

for handler in logging.getLogger().handlers:
    handler.addFilter(ScriptOnlyFilter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def run_one_by_one(n_runs: int = 1):
    errors_greedy = []
    errors_black = []

    for i in range(n_runs):
        # Run one GREEDY
        err_g = Main(greedy=True)
        errors_greedy.append(err_g)
        logger.debug(f"[GREEDY]    Run {i+1}/{n_runs} -> Error: {err_g}")
        logger.debug(f"[GREEDY]    Running mean: {np.mean(errors_greedy):.2f}")
        logger.debug(f"[BLACKLIST previus] Running mean: {np.mean(errors_black):.2f}")

        # Run one BLACKLIST
        err_b = Main(greedy=False)
        errors_black.append(err_b)
        logger.debug(f"[BLACKLIST] Run {i+1}/{n_runs} -> Error: {err_b}")
        logger.debug(f"[BLACKLIST] Running mean: {np.mean(errors_black):.2f}")
        logger.debug(f"[GREEDY]    Running mean: {np.mean(errors_greedy):.2f}")
        _, current_p = ttest_ind(np.array(errors_greedy), np.array(errors_black), equal_var=False)
        logger.debug(f"Current p-value: {current_p:.4f}")
        
        

    # Final stats
    greedy_arr = np.array(errors_greedy)
    black_arr = np.array(errors_black)

    mean_greedy = greedy_arr.mean()
    mean_black = black_arr.mean()
    absolute_improvement = mean_greedy - mean_black
    relative_improvement = (absolute_improvement / mean_greedy) * 100
    _, p_value = ttest_ind(greedy_arr, black_arr, equal_var=False)

    logger.debug("\n=== FINAL SUMMARY ===")
    logger.debug(f"[GREEDY]    Mean Error: {mean_greedy:.2f} | Errors: {errors_greedy}")
    logger.debug(f"[BLACKLIST] Mean Error: {mean_black:.2f} | Errors: {errors_black}")
    logger.debug(f"\nImprovement (BLACKLIST over GREEDY):")
    logger.debug(f"-> Absolute: {absolute_improvement:.2f}")
    logger.debug(f"-> Relative: {relative_improvement:.2f}%")
    logger.debug(f"-> p-value:  {p_value:.4f} (Welch’s t-test)")

if __name__ == "__main__":
    run_one_by_one(n_runs=45)
