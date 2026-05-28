# Author meta data:

__Name__ = "Syed Raza"
__Email__ = "sar0033@uah.edu"

# the import statements
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt

# make a Linear Regression class here based on torch
class LinearRegression:
    """
    A PyTorch implemented Linear Regression class 

    Model: y = X @ W + b
    Loss: Mean Squared Error

    Features:
    - Gradient based optimization using PyTorch
    """

    def __init__(self, no_features: int, learning_rate: float, 
                max_epochs: int, tolerance: float = 1e-6, 
                print_converge: str = False):
        """
        The constructor function for LinearRegression Class

        Params:
            - no_features, is the number of input variables
            - learning_rate, has to do with limiting the size of step taken with gradient descent
            - max_epochs, is the maximum number of epochs before the training stops 
            - tolerance, to know if things have converged
        """

        # class variables
        self.no_features = no_features
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.print_converge = print_converge

        # more class variables: related to data splitting 
        self.nsamples = None
        self.X_train = None
        self.y_train = None

        # a vraiable to see if the model has been trained
        self.trained = False

        # making weights here according to no_features
        self.W = nn.Parameter(torch.randn(no_features + 1, dtype=torch.float64, requires_grad=True))

        # initialize weights randomly
        nn.init.normal_(self.W, mean=0.0, std=0.01)

        # make loss (MSE) and optimier (Adam) here
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.SGD([self.W], lr=self.learning_rate)

        # to store loss history
        self.loss_history = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        The forward function for Linear Regression

        Params:
            - X, is the input data as a torch Tensor
        Returns:
            - the predicted values as a torch Tensor
        """
        # ensure X is float64 to match weights
        X = X.to(dtype=torch.float64)
        return self.W[0] + X @ self.W[1:]  # b + X @ W
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'LinearRegression':
        """
        The fit function for Linear Regression

        Params:
            - X_train, is the training input data as a numpy array
            - y_train, is the training labels as a numpy array
        Returns:
            - the trained LinearRegression object
        """

        # convert to pytorch tensors
        self.X_train = torch.tensor(X_train, dtype=torch.float64)
        self.y_train = torch.tensor(y_train, dtype=torch.float64)
        self.nsamples = len(X_train)

        prev_loss = float("inf")
        self.loss_history.clear()

        # the training loop
        for epoch in range(self.max_epochs):
            # reset the gradients 
            self.optimizer.zero_grad()

            # premature predictions
            pred = self.forward(self.X_train)

            # loss function
            loss = self.loss_function(pred, self.y_train)

            # automatic gradient backward pass
            loss.backward()

            # update the model parameters
            self.optimizer.step()

            # save current loss
            current_loss = float(loss.detach().item())

            # save loss in history
            self.loss_history.append(current_loss)

            # convergence check
            if abs(prev_loss - current_loss) < self.tolerance:
                if self.print_converge == True:            
                    print(f"Converged after {epoch + 1} epochs")
                break

            prev_loss = current_loss
        
        # model is now trained 
        self.trained = True
        return self 
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        The predict function for Linear Regression

        Params:
            - X_test, is the test input data as a numpy array
        Returns:
            - the predicted values as a numpy array
        """
        if not self.trained:
            warnings.warn("Model has not been trained yet. Pre64ctions may be unreliable.", UserWarning)

        # convert to pytorch tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float64)

        # get predictions
        with torch.no_grad():
            predictions = self.forward(X_test_tensor)

        return predictions.numpy()

    # make a plot for loss history
    def loss_history_plot(self):
        """
        The line plot loss_history_plot function for Linear Regression

        Plots the loss history of the trained model
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before plotting loss history.")

        # line plot for loss history
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=range(1, len(self.loss_history) + 1), y=self.loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.title("Loss History over Epochs")
        plt.yscale("log")
        plt.show()

    # a function to plot predictions vs actuals
    def predictions_vs_actuals_plot(self, X_test: np.ndarray, y_test: np.ndarray, target: str, save_path: str) -> 'LinearRegression':
        """
        The scatter plot predictions_vs_actuals_plot function for Linear Regression

        Plots the predicted values vs actual values of the trained model with a gradient overlay.
        Points closer to the perfect prediction line are bright; farther ones are darker.
        
        Params:
            - cmap: colormap name (default: 'coolwarm'). Other options: 'viridis', 'plasma', 'RdYlGn', etc.
        """

        # check if model is trained
        if not self.trained:
            raise RuntimeError("Model must be trained before plotting predictions vs actuals.")
        
        # get predictions and residuals
        predictions = self.predict(X_test)
        residuals = np.abs(y_test - predictions)
        
        # normalize the residuals for color mapping 
        norm_residuals = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-8)

        # scatter plot for predicted vs actuals with gradient coloring
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(y_test, predictions, c=norm_residuals, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel(f"{target} observed (nT)", fontsize=14)
        plt.ylabel(f"{target} predicted (nT)", fontsize=14)
        plt.title(f"Predicted vs Observed {target} with ADLR", fontsize=16)
        plt.colorbar(scatter, label='Normalized Error')
        plt.legend()
        # save the figure with dpi = 300 and tight layout
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()