PYTHON = python
NAMES = HB AN TW RB

# Allow `make run HB` style: treat extra goals as the NAME and stub them out.
EXTRA_GOALS := $(filter-out run check-env clean,$(MAKECMDGOALS))
ifneq ($(EXTRA_GOALS),)
NAME ?= $(firstword $(EXTRA_GOALS))
endif

# check that torch is importable in the current environment
check-env:
	@$(PYTHON) -c "import torch" 2>/dev/null || (echo "Error: torch not found. Activate the correct virtual environment." && exit 1)

run: check-env
	@if [ -n "$(NAME)" ]; then \
		TARGETS="$(NAME)"; \
	else \
		TARGETS="$(NAMES)"; \
	fi; \
	for name in $$TARGETS; do \
		echo "=== Running $$name ==="; \
		$(PYTHON) train.py --name $$name && \
		$(PYTHON) inference.py --name $$name && \
		$(PYTHON) plot_result.py --name $$name; \
	done

# Swallow extra goals like `make run HB` so they don't error out.
$(EXTRA_GOALS):
	@:

clean:
	rm -f flow_*.pt data_*.pt result_*.png
	rm -rf __pycache__

.PHONY: run clean check-env $(EXTRA_GOALS)
