import pytest
import torch
from src.training.lora_model import MockLinearWithLoRA
from src.training.pipeline import CustomTrainingLoopPipeline

def get_test_device():
    """動態感知當前測試環境最優可用裝置"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

def test_lora_weights_initialization_delta_zero():
    """【測試目標】驗證工業標準：初始化時 LoRA B 必須全為 0，此時輸出必須等價於 Base Model"""
    model = MockLinearWithLoRA(in_features=10, out_features=5, r=4, alpha=16)
    mock_input = torch.randn(2, 10)

    with torch.no_grad():
        actual_output = model(mock_input)
        expected_base_output = torch.matmul(mock_input, model.base_weight.data.t())

        # 斷言：初始狀態下 LoRA 旁路不應造成任何干擾
        assert torch.allclose(actual_output, expected_base_output, atol=1e-5)

def test_gradient_convergence_and_accumulation():
    """【測試目標】驗證自製 Custom Loop 在任何平台上皆具備使 Loss 收斂與正確執行累積更新的能力"""
    torch.manual_seed(42)
    device = get_test_device()

    model = MockLinearWithLoRA(in_features=8, out_features=4, r=2, alpha=8)
    pipeline = CustomTrainingLoopPipeline(model, lr=1e-2, grad_accum_steps=2, max_grad_norm=1.0, device=device)

    # 建立過擬合模擬數據
    X = torch.randn(10, 8)
    Y = torch.randn(10, 4)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    loss_epoch_1 = pipeline.train_epoch(dataloader, epoch_idx=1)
    loss_epoch_2 = pipeline.train_epoch(dataloader, epoch_idx=2)

    print(f"跨平台 TDD Loss 進程觀測 ({device}): 輪次1={loss_epoch_1:.4f}, 輪次2={loss_epoch_2:.4f}")

    # 斷言：優化器正確運作下，第二輪平均 Loss 必須下降
    assert loss_epoch_2 < loss_epoch_1

def test_merge_and_unload_mathematical_identity():
    """【測試目標】驗證封裝後的 merge_and_unload 內部熔斷完全符合推理恆等性"""
    model = MockLinearWithLoRA(in_features=16, out_features=8, r=4, alpha=16)

    # 人為賦予 LoRA 矩陣隨機權重（模擬微調完畢狀態）
    model.lora_A.data = torch.randn(4, 16)
    model.lora_B.data = torch.randn(8, 4)

    mock_inference_input = torch.randn(3, 16)

    model.eval()
    with torch.no_grad():
        # 1. 取得合併前，雙軌並行查表的推理輸出
        output_before_merge = model(mock_inference_input)

    # 2. 🌟 呼叫全新高階封裝的內部熔斷函式（內部自動歸零旁路）
    model.merge_and_unload()

    # 3. 取得合併後，旁路已銷毀、僅剩主幹道全速前進的推理輸出
    with torch.no_grad():
        output_after_merge = model(mock_inference_input)

    # 斷言：消除雙軌旁路結構前後，最終輸出的數學結果必須完美契合，絕無精度位移
    assert torch.allclose(output_before_merge, output_after_merge, atol=1e-4)