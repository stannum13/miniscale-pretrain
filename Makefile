.PHONY: smoke single-gpu distributed benchmark fault-test report test

smoke:
	python train.py --config configs/smoke.yaml --output-dir results/smoke --run-id "smoke-$$(date -u +%Y%m%dT%H%M%SZ)"

single-gpu:
	scripts/single_gpu.sh

distributed:
	scripts/distributed.sh

benchmark:
	scripts/benchmark.sh

fault-test:
	python scripts/fault_test.py

report:
	python scripts/render_report.py --results results --output REPORT.md

test:
	pytest -q
