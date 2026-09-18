.PHONY: smoke correctness single-gpu distributed benchmark fault-test report test \
	cloud-status cloud-bucket cloud-launch cloud-logs cloud-download cloud-delete

DATASET_REVISION ?= 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
TOKENIZER_REVISION ?= 93efa2f097d58c2a74874c7e644dbc9b0cee75a2
CLOUD_RUNNER = python scripts/gcloud_runner.py

smoke:
	python train.py --config configs/smoke.yaml --output-dir results/smoke --run-id "smoke-$$(date -u +%Y%m%dT%H%M%SZ)"

correctness:
	python scripts/check_loss_agreement.py

single-gpu:
	scripts/single_gpu.sh

distributed:
	scripts/distributed.sh

benchmark:
	scripts/benchmark.sh

fault-test:
	python scripts/fault_test.py --world-size 1
	python scripts/fault_test.py --world-size 2

report:
	python scripts/render_report.py --results results --output REPORT.md

test:
	pytest -q

cloud-status:
	$(CLOUD_RUNNER) status

cloud-bucket:
	$(CLOUD_RUNNER) bucket

cloud-launch: cloud-bucket
	$(CLOUD_RUNNER) create \
		--source-commit="$$(git rev-parse HEAD)" \
		--dataset-revision="$(DATASET_REVISION)" \
		--tokenizer-revision="$(TOKENIZER_REVISION)"

cloud-logs:
	$(CLOUD_RUNNER) logs

cloud-download:
	$(CLOUD_RUNNER) download

cloud-delete:
	$(CLOUD_RUNNER) delete
