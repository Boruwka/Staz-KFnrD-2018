import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

l_miast = 9
l_miesiecy = 12
factors = 2

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
cities = ["Toronto", "Warsaw", "Boston", "London", "San Francisco", "Jerusalem", "Mexico", "Cape Town", "Sydney"]
avg_temp = np.array([
    [-5.8, -3.1, 4.5, 6.7, 14.3, 18.2, 20.1, 20.6, 15.9, 11.2, 3.6, -7.2],
    [-2.9, 3.6, 4.2, 9.7, 16.1, 19.5, 20.0, 18.8, 16.4, 7.6, 3.2, 1.3],
    [0.3, 1.5, 5.9, 8.4, 14.8, 20.2, 24.5, 24.7, 19.7, 13.0, 7.9, 1.9],
    [2.3, 6.5, 8.7, 9.2, 12.3, 15.4, 17.3, 20.0, 14.8, 10.8, 8.7, 6.4],
    [11.5, 13.9, 14.3, 15.7, 16.3, 17.4, 17.2, 17.7, 18.2, 17.4, 14.6, 10.4],
    [9.7, 10.3, 12.7, 15.5, 21.2, 22.1, 24.1, 25.3, 23.5, 20.1, 15.7, 11.8],
    [14.0, 15.6, 17.5, 20.3, 20.6, 18.1, 17.6, 18.2, 17.8, 16.8, 14.9, 16.0],
    [23.1, 23.3, 21.4, 19.0, 17.1, 15.5, 15.4, 15.6, 15.4, 18.6, 20.9, 21.3],
    [23.8, 24.6, 23.4, 20.8, 18.1, 15.1, 14.4, 14.5, 17.3, 19.0, 21.8, 24.3]
])

df = pd.DataFrame(avg_temp, index=cities, columns=months)
#sns.heatmap(df, annot=True, fmt='.0f')
df.values.shape
class Factorize(nn.Module):
    
    def __init__(self, factors):
        super(Factorize, self).__init__()
        self.A = Parameter(torch.randn(l_miast, factors))
        self.B = Parameter(torch.randn(factors, l_miesiecy))
	self.global_bias = Parameter(torch.randn(1))
	self.bias_miast = Parameter(torch.randn(l_miast))
    
    def forward(self):
        output = self.A.matmul(self.B)+self.global_bias
	output = output.transpose(0,1)
	for i in range(l_miesiecy):
		output[i] = output[i]+self.bias_miast
	output = output.transpose(0,1)
        return output
model = Factorize(factors)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
list(model.parameters())
desired_output = Variable(torch.FloatTensor(df.values))

loss2 = []
for i in range(10000):
    optimizer.zero_grad()
    output = model()
    loss = criterion(output, desired_output)
    loss2.append(loss.item())
    loss.backward()        
    optimizer.step()

print("Loss at the end: {:.2f}".format(loss2[-1]))
#plt.plot(range(len(loss2)), loss2)
#plt.show()
#plt.close()

A = model.A.transpose(0,1).detach().numpy()
print A
sns.regplot(x=A[0], y=A[1], fit_reg=False).set_title('Miasta')
plt.show()
plt.close()
B = model.B.detach().numpy()
print B
sns.regplot(x=B[0], y=B[1], fit_reg=False).set_title('Miesiace')
plt.show()
plt.close()


avg_temp_pred = output.detach().numpy()
df2 = pd.DataFrame(avg_temp_pred, index=cities, columns=months)

#sns.heatmap(df2, annot=True, fmt='.0f')
#plt.show()
#plt.close()

# differences
#sns.heatmap(df2 - df, annot=True, fmt='.0f')
#plt.show()
#plt.close()
