from typing import Tuple
# from abc import ABCMeta, abstractmethod
# from ..scalers import MinMaxScaler, StandardScaler
# from .gpytorch_settings import gpytorch_settings

import gpytorch
import numpy as np
import torch

class IndependentBGKIModel():
    default_evidence = 1e-2
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        kernel: gpytorch.kernels.Kernel,
        lr: float = 1e-2,
        batch_size: int = 128, #mini-BatchSGD used
        jitter: float = 1e-6,    
    ):
        self.train_x = self._preprocess_x(x_train)
        self.train_y = self._preprocess_y(y_train)
        self.train_sigma2 = torch.ones(self.train_y.shape[0], dtype=torch.float64) * self.default_evidence
        self.kernel = kernel.double()# convert dtype to float64, more precise
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.kernel.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.jitter = jitter

        # print(f'\n======================================== IndependentBGKIModel Init debug ========================================\n')
        # print(f"train_x shape: {self.train_x.shape}")
        # print(f"train_y shape: {self.train_y.shape}")
        # for name, param in kernel.named_parameters():
        #     print(f"Parameter name: {name} | requires_grad: {param.requires_grad}")
        #     print(f"Shape: {param.shape}")
        #     print(f"Values :\n{param.data}\n")


    def _preprocess_x(self, x: np.ndarray) -> torch.Tensor:
        x = torch.tensor(x, dtype=torch.float64)
        return x

    def _preprocess_y(self, y: np.ndarray) -> torch.Tensor:
        y = torch.tensor(y, dtype=torch.float64).squeeze(-1)
        return y
    
    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        self.train_x = torch.cat((self.train_x, self._preprocess_x(x)), dim=0)
        self.train_y = torch.cat((self.train_y, self._preprocess_y(y)), dim=0)
        # self.train_sigma2 = torch.cat((self.train_sigma2, torch.ones_like(self._preprocess_y(y)) * self.default_evidence), dim=0)
        self.train_sigma2 = torch.cat((self.train_sigma2, torch.ones(self._preprocess_y(y).shape[0], dtype=torch.float64) * self.default_evidence), dim=0)
        # print(f'\n======================================== After adding data: ========================================\n')
        # print(f"train_x shape: {self.train_x.shape}")
        # print(f"train_y shape: {self.train_y.shape}")
        # print(f"train_sigma2 shape: {self.train_sigma2.shape}")


    def optimize(self, num_steps: int):
        # print(f'\n======================================== IndependentBGKIModel Optimation Debug: ========================================')
        losses = []
        content = []
        for i in range(num_steps):
            self.optimizer.zero_grad()

            batch = torch.randint(high=len(self.train_y), size=(self.batch_size,))
            batch_x = self.train_x[batch]
            batch_y = self.train_y[batch]
            batch_sigma2 = self.train_sigma2[batch]

            # compute loss
            #！ LOO predict， getting mu and sigma2
            kernel_cov = self.kernel(batch_x, batch_x)
            kernel_cov = kernel_cov.evaluate()
            kbar_loo = kernel_cov.sum(dim=1) + 1.0 / (batch_sigma2 + self.jitter) - kernel_cov.diag()
            ybar_loo = kernel_cov @ batch_y + batch_y / (batch_sigma2 + self.jitter).unsqueeze(-1) - kernel_cov.diag().diag_embed() @ batch_y
            mu = ybar_loo / kbar_loo.unsqueeze(-1)
            sigma2 = 1 / kbar_loo

            #! compute negative log likelihood loss
            sigma2_all = sigma2 + batch_sigma2
            term0 = batch_x.shape[0] * np.log(2*np.pi)
            term1 = torch.log(sigma2_all)
            term2 = (batch_y - mu)**2 / sigma2_all.unsqueeze(-1)
            loss = 0.5 * (term0 + term1.unsqueeze(-1) + term2).sum()
            loss *= 1000

            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()

            # print(f"Iter {i:04d} | NLL-LOSS: {loss.item(): .2f} ")

        #     #！！It was updated!!!
        #     print(f'\n===== Optimizer Updated over')
        #     print(f"train_x shape: {self.train_x.shape}")
        #     print(f"train_y shape: {self.train_y.shape}")
        #     for name, param in self.kernel.named_parameters():
        #         print(f"Parameter name: {name} | requires_grad: {param.requires_grad}")
        #         print(f"Shape: {param.shape}")
        #         print(f"Values :\n{param.data}\n")

        #     # 定义要写入文件的内容
            
        #     content.append(f'\n===== Optimizer Updated over at step {i}')
        #     content.append(f"train_x shape: {self.train_x.shape}")
        #     content.append(f"train_y shape: {self.train_y.shape}")
        #     for name, param in self.kernel.named_parameters():
        #         content.append(f"Parameter name: {name} | requires_grad: {param.requires_grad}")
        #         content.append(f"Shape: {param.shape}")
        #         # 将 param.data 转换为字符串并追加到 content 中
        #         param_data_str = str(param.data)
        #         # 格式化param_data_str的输出，控制小数点后位数
        #         param_data_list = []
        #         for line in param_data_str.split('\n'):
        #             if line.strip().startswith('tensor'):
        #                 param_data_list.append(line)
        #             elif line.strip().startswith('...'):
        #                 param_data_list.append(line)
        #             else:
        #                 line = line.replace('[', '').replace(']', '').strip()
        #                 if line:
        #                     values = line.split()
        #                     formatted_values = ['{:.4f}'.format(float(v)) for v in values]
        #                     param_data_list.append('[' + ' '.join(formatted_values) + ']')
        #         param_data_str_formatted = '\n'.join(param_data_list)
        #         content.append(f"Values :\n{param_data_str_formatted}\n")

        # # 将内容写入文件
        # with open("optimizer_update.txt", "w") as file:
        #     for line in content:
        #         file.write(line + "\n")
        return losses

    def predict(
            self, 
            predX: np.ndarray, 
            predYPrior: np.ndarray = None, 
            predYSigma2: np.ndarray = 1e5) -> Tuple[np.ndarray, np.ndarray]:
        predX_tensor = self._preprocess_x(predX)
        kernel = self.kernel(predX_tensor, self.train_x)
        if predYPrior is None:
            predYPrior = torch.zeros([predX_tensor.shape[0]], dtype=torch.float64) 
        ybar = kernel @ self.train_y + predYPrior/ (predYSigma2 + self.jitter)
        kbar = kernel.sum(dim=1) + 1 / (predYSigma2 + self.jitter)
        mu = ybar / (kbar + self.jitter)
        Sigma2 = 1 / kbar
        # print(f'YCY-DEBUG')
        # print(f'mu.shape:{mu.shape}')
        # print(f'Sigma2.shape:{Sigma2.shape}')
        if mu.ndim == 1:
            mu = mu.unsqueeze(-1)
        if Sigma2.ndim == 1:
            Sigma2 = Sigma2.unsqueeze(-1)
        return mu.cpu().detach().numpy(), Sigma2.cpu().detach().numpy()
    
    def get_bgki_lengthscales(self):
        lengthscales = []
        for name, param in self.kernel.named_parameters():
            if 'kLenMat_' in name:
                lengthscales.append(param.data)
        if len(lengthscales) == 0:
            raise ValueError("No lengthscale found in the kernel parameters.")
        else:
            return torch.cat(lengthscales, dim=0).cpu().detach().numpy()


        
