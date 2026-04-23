PYTHON = python
NAME = AN

# check that torch is importable in the current environment
check-env:
	@$(PYTHON) -c "import torch" 2>/dev/null || (echo "Error: torch not found. Activate the correct virtual environment." && exit 1)

run: check-env train infer plot

train: check-env
	$(PYTHON) train.py --name $(NAME)

infer: check-env
	$(PYTHON) inference.py --name $(NAME)

plot: check-env
	$(PYTHON) plot_result.py --name $(NAME)

test: check-env
	$(PYTHON) potential.py
	$(PYTHON) -m pytest utilities.py 2>/dev/null || $(PYTHON) utilities.py

clean:
	rm -f flow_*.pt data_*.pt result_*.png
	rm -rf __pycache__

.PHONY: run train infer plot test clean check-env
