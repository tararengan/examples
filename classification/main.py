#A binary classifier trained on a neural network using pytorch


#imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import sys


#constants


def generate_points(size = 50):

    N = size
    c1_points = np.multiply(np.random.choice([1, -1], size=(N, 1)),
                         np.random.randint(4, 10, (N, 1), 'int32')  + np.random.randn(N, 1))
    c2_points = np.multiply(np.random.choice([1, -1], size=(N, 1)),
                            np.random.randint(3, 10, (N, 1), 'int32') + np.random.randn(N, 1))

    points = np.concatenate((c1_points, c2_points), axis=1).reshape(N, 2)

    return points


def generate_labels_by_quad(points):
    N = points.shape[0]
    labels = np.ones((N,))

    c1_points =  points[:, 0].reshape(N,)
    c2_points = points[:, 1].reshape(N,)

    indices_q1 = np.where((c1_points > 0) & (c2_points > 0))
    indices_q2 = np.where((c1_points <= 0) & (c2_points > 0))
    indices_q3 = np.where((c1_points <= 0) & (c2_points <= 0))
    indices_q4 = np.where((c1_points > 0) & (c2_points <= 0))

    labels[indices_q1[0]] = 0
    labels[indices_q2[0]] = 1
    labels[indices_q3[0]] = 2
    labels[indices_q4[0]] = 3

    x_ = points
    y_ = labels

    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).long()

    # print(type(y), y.size(), y.dtype)

    return x_, y_


def generate_labels_by_rot_quad(points):
    N = points.shape[0]
    labels = np.ones((N,))

    c1_points = points[:, 0].reshape(N, 1)
    c2_points = points[:, 1].reshape(N, 1)

    indices_q1 = np.where((c2_points > 0) & (abs(c1_points) < abs(c2_points)))
    indices_q2 = np.where((c1_points <= 0) & (abs(c2_points) <= abs(c1_points)))
    indices_q3 = np.where((c2_points < 0) & (abs(c1_points) < abs(c2_points)))
    indices_q4 = np.where((c1_points > 0) & (abs(c2_points) <= abs(c1_points)))

    labels[indices_q1[0]] = 0
    labels[indices_q2[0]] = 1
    labels[indices_q3[0]] = 2
    labels[indices_q4[0]] = 3

    x_ = points
    y_ = labels

    x_ = torch.from_numpy(x_).float()
    y_ = torch.from_numpy(y_).long()

    # print(type(y), y.size(), y.dtype)

    return x_, y_


def generate_data(batch_size =50):
    points = generate_points(batch_size)
    x, y = generate_labels_by_rot_quad(points)

    return x, y


def get_batch(batch_size=50):
    x, y = generate_data(batch_size)
    return x, y


#define model - takes in features and returns log probabilities
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(2, 2)
        self.lin2 = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.logprob = nn.LogSoftmax()

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.activation(x).squeeze()
        #x = self.logprob(x)
        return x


net = Net()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.002, momentum=.8)

#get data in batches
x_data = []
y_data = []
num_batches = 15
batch_size = 20
for i in range(num_batches):
    x, y = get_batch(batch_size)
    x_data.append(x)
    y_data.append(y)




predictions = []
for epoch in range(40):

    running_loss = 0.0

    for i in range(num_batches):

        inputs = x_data[i]
        labels = y_data[i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(type(outputs), outputs.size(), outputs.dtype)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:  # print every r mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0


print('Finished training \n')

#get predictions and accuracy
plt.title('A 2D Classifier')
plt.ion()
plt.show()
accuracy = 0
for i in range(num_batches):

    inputs = x_data[i]
    labels = y_data[i]

    outputs = net(inputs)
    max_values, predictions = torch.max(outputs, 1)

    print(labels.numpy(), predictions.numpy())
    # print(len(np.where(labels.numpy() == predictions.numpy())[0]))
    accuracy += len(np.where(labels.numpy() == predictions.numpy())[0])

    class_1_indices = list(np.where(predictions.numpy() == 0)[0])
    class_2_indices = list(np.where(predictions.numpy() == 1)[0])
    class_3_indices = list(np.where(predictions.numpy() == 2)[0])
    class_4_indices = list(np.where(predictions.numpy() == 3)[0])

    coord_cl1 = x_data[i].numpy()[class_1_indices, :]
    coord_cl2 = x_data[i].numpy()[class_2_indices, :]
    coord_cl3 = x_data[i].numpy()[class_3_indices, :]
    coord_cl4 = x_data[i].numpy()[class_4_indices, :]

    if i == 0:
        coord_values_cl1 = coord_cl1
        coord_values_cl2 = coord_cl2
        coord_values_cl3 = coord_cl3
        coord_values_cl4 = coord_cl4
    else:
        coord_values_cl1 = np.concatenate((coord_values_cl1, coord_cl1), axis=0)
        coord_values_cl2 = np.concatenate((coord_values_cl2, coord_cl2), axis=0)
        coord_values_cl3 = np.concatenate((coord_values_cl3, coord_cl3), axis=0)
        coord_values_cl4 = np.concatenate((coord_values_cl4, coord_cl4), axis=0)


accuracy = accuracy/(batch_size*num_batches)
print('Accuracy: {0}'.format(accuracy))
print(net.lin1._parameters)
print(net.lin2._parameters)

plt.scatter(coord_values_cl1[:, 0], coord_values_cl1[:, 1], color='r', s=10)
plt.scatter(coord_values_cl2[:, 0], coord_values_cl2[:, 1], color='b', s=10)
plt.scatter(coord_values_cl3[:, 0], coord_values_cl3[:, 1], color='c', s=10)
plt.scatter(coord_values_cl4[:, 0], coord_values_cl4[:, 1], color='y', s=10)


plt.pause(.001)
input('Press enter to continue')



















