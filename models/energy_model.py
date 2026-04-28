
class EnergyPredictor:
    def __init__(self, model_tox, w1, model_strain=None, w2=0,model_aff=None,w3=0,
                 static_tox_scale=1,static_strain_scale=1.16):

        self.model_tox = model_tox
        self.w1 = w1
        self.model_strain = model_strain
        self.w2 = w2
        self.static_tox_scale = static_tox_scale  # 保存因子
        self.model_aff = model_aff
        self.w3 = w3
        self.static_strain_scale = static_strain_scale

        print("EnergyPredictor initialized.")
        print(f"  - Tox Model: {'Loaded' if model_tox else 'None'}")
        if model_tox:
            print(f"  - [!!!] Tox E/Grad is STATICALLY SCALED by {self.static_tox_scale:.2f}")
            print(f"  - [!!!] Strain E/Grad is STATICALLY SCALED by {self.static_strain_scale:.2f}")
        print(f"  - Strain Model: {'Loaded' if model_strain else 'None'}")

    def predict(self, batch, differentiable=True):

        E_total = 0.0

        # 1. 计算“缩放后”的毒性
        if self.model_tox is not None and self.w1 != 0:
            E_tox_original = self.model_tox.predict(
                batch,
                differentiable=differentiable
            )
            # [!!!] 在求和前，将能量乘以缩放因子 [!!!]
            E_tox_scaled = E_tox_original * self.static_tox_scale
            E_total = E_total + (self.w1 * E_tox_scaled)
        if self.model_strain is not None and self.w2 != 0:
            E_strain_original = self.model_strain.predict(
                batch,
                differentiable=differentiable
            )
            E_strain_scaled = E_strain_original * self.static_strain_scale
            E_total = E_total + (self.w2 * E_strain_scaled)
        if self.model_aff is not None and self.w3 != 0:
            E_aff_original = self.model_aff.predict(
                batch,
                differentiable=differentiable
            )
            E_total = E_total - (self.w3 * E_aff_original)

        return E_total
