from scripts.fault_test import build_train_command


def test_distributed_fault_command_uses_loopback_torchrun() -> None:
    command = build_train_command(("--config", "configs/smoke.yaml"), world_size=2, port=29601)
    assert command[:2] == ["torchrun", "--nnodes=1"]
    assert "--nproc-per-node=2" in command
    assert "--master-addr=127.0.0.1" in command
    assert command[-2:] == ["--config", "configs/smoke.yaml"]
