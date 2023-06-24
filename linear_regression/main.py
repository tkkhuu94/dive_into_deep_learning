import torch

import data
import model
import trainer

def main():
    w = torch.tensor([2, -3.4])
    b = 4.2
    regression_data = data.SyntheticRegressionData(
        w=w,
        b=b,
        std_dev=0.01,
        num_train=1000,
        num_val=1000,
        batch_size=32,
    )

    regression_model = model.LinearRegression(2, lr=0.03)
    print('Model before train', regression_model)

    regression_trainer = trainer.Trainer(max_epochs=5)
    regression_trainer.fit(regression_model, regression_data)

    print('Ground truth', w, b)
    print('Model after train', regression_model)
    print(f'error in estimating w: {regression_data.W - regression_model.w.reshape(regression_data.W.shape)}')
    print(f'error in estimating b: {regression_data.b - regression_model.b}')

if __name__ == "__main__":
    main()