import torch
import torch.nn as nn
import os

class MockLinearWithLoRA(nn.Module):
    """
    自研生產級、平台中立之擬真 LoRA 封裝層。
    【架構承諾】所有降維與還原熔斷之矩陣運算完全內聚於本類別，對外僅暴露高階語意接口。
    """
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # ----------------------------------------------------------------------
        # 步驟一：配置唯讀常駐區（封鎖寫入緩衝區，防範本機記憶體爆炸）
        # ----------------------------------------------------------------------
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        # ----------------------------------------------------------------------
        # 步驟二：配置活躍增量小表（負責降維與還原核心）
        # 依據工業標準：A 採用高斯初始化，B 初始化為全 0，確保起跑點 \Delta W = 0
        # ----------------------------------------------------------------------
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02, requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        【步驟三：執行內部降維傳送帶計算】
        主流程只丟入資料，內部自動分流查表、壓縮並疊加，主流程對此完全無感知。
        """
        # 主幹道：唯讀大庫矩陣乘法計算（提取基礎通用智慧）
        base_out = torch.matmul(x, self.base_weight.t())

        # 旁路雙軌：lora_A(高維特徵壓縮降到8維) -> lora_B(8維低維特徵還原回高維) -> 放大訊號強度
        lora_out = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scaling

        return base_out + lora_out

    def merge_and_unload(self) -> None:
        r"""
        【內部熔斷還原演算法】
        將 \Delta W 永久性熔斷融入 base_weight，並物理性清空旁路，杜絕重複加法污染與線上推理延遲。
        """
        with torch.no_grad(): # 顯式宣示關閉計算圖，最大化記憶體釋放效率
            # 1. 記憶體內部計算完成還原的變更矩陣 (Delta W = B @ A * scaling)
            delta_W = torch.matmul(self.lora_B.data, self.lora_A.data) * self.scaling

            # 2. 原生 Inplace 加法操作，直接覆寫進大模型的唯讀常駐空間中
            self.base_weight.data.add_(delta_W)

            # 3. 🌟 狀態防禦核心：物理清空旁路權重，防範 forward 發生二次重複疊加
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()
            self.scaling = 0.0  # 強制將旁路訊號增益阻斷歸零

            print("⚔️  [Model Internal] 內部熔斷合併完工！增量已同步覆寫，旁路快取已安全銷毀。")

    def save_lora_weights(self, folder_path: str, filename: str = "lora_weights.pt"):
        """【步驟五：執行內部資產過濾儲存】僅將輕量小表序列化存盤，全面確保跨平台相容性"""
        os.makedirs(folder_path, exist_ok=True)
        target_path = os.path.join(folder_path, filename)
        state_dict = {
            "lora_A": self.lora_A.data.cpu(),  # 強制拉回 CPU 抹除裝置憑證，防止異質反序列化崩潰
            "lora_B": self.lora_B.data.cpu(),
            "r": self.r,
            "alpha": self.alpha
        }
        torch.save(state_dict, target_path)
        print(f"💾 [Model Internal] 輕量增量結晶檔案已安全倒出: {target_path}")